import os
import base64
import copy
import json
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from typing import List
from tqdm import tqdm

import pydub
import pandas as pd
import numpy as np
from pymongo import MongoClient

import whisper
import torch

from help_function import read_json, save_json, compute_euclidean_distance


DATA_PATH = "exp_data/export"
EXPERIMENT_NAME = "Exp1"


def experiment_to_simulation(experiment_to_timeline, simulations):
    for experiment_i in tqdm(experiment_to_timeline, desc=f"Simulation to timeline"):
        timeline_i = experiment_to_timeline[experiment_i]
        for simulation in simulations:
            for player_id in simulations[simulation]["player"]:
                player_time = simulations[simulation]["player"][player_id]
                time_dif_1 = timeline_i - player_time
                time_dif_2 = player_time - timeline_i
                if (time_dif_1.days == 0 and time_dif_1.seconds <= 3600) or (
                        time_dif_2.days == 0 and time_dif_2.seconds <= 3600):
                    simulations[simulation]["experiment"] = experiment_i
                    break


def pandas_read_questionary_csv(data_dir: str, values_dir: str, variables_dir: str, experiment_participant: List[int]):
    df = pd.read_csv(data_dir, encoding="utf-16", sep="\t", header=0, skiprows=[1])
    df_values = pd.read_csv(values_dir, encoding="utf-16", sep="\t", header=0)
    df_variables = pd.read_csv(variables_dir, encoding="utf-16", sep="\t", header=0)

    counter = 1
    experiment_counter = 1
    simulation_pointer = 0
    df.columns.values.tolist()
    sum_participants = experiment_participant[0]
    data_json = json.loads(df.to_json(orient="records"))
    values_json = json.loads(df_values.to_json(orient="records"))
    variables_json = json.loads(df_variables.to_json(orient="records"))
    values_json_var_format = {}

    for i in values_json:
        if i["VAR"] not in values_json_var_format:
            values_json_var_format[i["VAR"]] = {}
            values_json_var_format[i["VAR"]]["RESPONSES"] = {}
        if i["RESPONSE"] not in values_json_var_format[i["VAR"]]["RESPONSES"]:
            values_json_var_format[i["VAR"]]["RESPONSES"][i["RESPONSE"]] = i["MEANING"]
    for i in variables_json:
        if i["VAR"] in values_json_var_format:
            values_json_var_format[i["VAR"]]["LABEL"] = i["LABEL"]
            values_json_var_format[i["VAR"]]["QUESTION"] = i["QUESTION"]
            values_json_var_format[i["VAR"]]["INPUT"] = i["INPUT"]
        elif i["INPUT"] != "SYSTEM":
            values_json_var_format[i["VAR"]] = i

    simulation_participants = {}

    sorted_after_experiment = sorted(data_json, key=lambda x: x['DI03_01'])
    for participant_i in sorted_after_experiment:
        if counter > sum_participants:
            simulation_pointer += 1
            sum_participants += experiment_participant[simulation_pointer]
            experiment_counter += 1
        if experiment_counter not in simulation_participants:
            simulation_participants[experiment_counter] = {}
        simulation_participants[experiment_counter][counter] = participant_i
        counter += 1
        print(participant_i)

    myKeys = list(values_json_var_format.keys())
    myKeys.sort()
    values_json_var_format = {i: values_json_var_format[i] for i in myKeys}
    save_json(simulation_participants, f"{DATA_PATH}/all_simulation_evaluation_infos.json")
    save_json(values_json_var_format, f"{DATA_PATH}/evaluation_values_variables.json")
    return simulation_participants


class PyMongoConnect:
    def __init__(self, config_data_dir: str):
        self.config_data = read_json(config_data_dir)
        db_name = self.config_data["Datenbank"]
        self.client = MongoClient(
            f'mongodb://{quote_plus(self.config_data["User"])}:{quote_plus(self.config_data["Passwort"])}@{self.config_data["Server"]}:{self.config_data["Port"]}')
        self.db = self.client[db_name]

    def get_role_player(self, simulations):
        db_role = self.db["Role"]
        for player in tqdm(simulations, desc="Get Roles of all Players"):
            time_log_in = player["serverTime"]
            all_roles = db_role.find({"playerId": player["playerId"]})
            for role_i in all_roles:
                time_i = role_i["serverTime"]
                time_dif = time_i - time_log_in
                if time_dif.days == 0 and time_dif.seconds <= 3600:
                    player["role"] = role_i["role"]
                    player["roleLogIn"] = time_i
                    break
        return simulations

    def get_all_specific_data_players(self, simulations, list_collections: List[str], ):
        for player in simulations:
            player_time = player["serverTime"]
            player_id = player["playerId"]
            if "role" not in player:
                continue
            role = player["role"]

            player_time_str = player_time.strftime('%Y-%m-%d_%H-%M-%S.%f')

            for collection_i in tqdm(list_collections, desc=f"Get all Information of chosen collections for Player:{player_id}"):
                db_collection = self.db[collection_i]
                results = db_collection.find({"playerId": player_id})
                all_outputs = []
                for c, result_i in enumerate(results):
                    result_i.pop("_id")
                    if "serverTime" in result_i:
                        result_i["serverTime"] = result_i["serverTime"].strftime('%Y-%m-%d %H:%M:%S.%f')
                    else:
                        result_i["server-timestamp"] = result_i["server-timestamp"].strftime('%Y-%m-%d %H:%M:%S.%f')

                    all_outputs.append(result_i)
                    player[collection_i] = all_outputs

            player["serverTime"] = player["serverTime"].strftime('%Y-%m-%d %H:%M:%S.%f')
            player["roleLogIn"] = player["roleLogIn"].strftime('%Y-%m-%d %H:%M:%S.%f')
            player.pop("_id")
            save_json(player, f"{DATA_PATH}/player_data/{player_time_str}_{role}_{player_id}_data.json")
        return simulations


    def format_datetimes(self, simulations, list_collections: List[str]):
        for sim in tqdm(simulations, desc="Transforms all datetime into string"):
            for player_id in simulations[sim]["players"]:
                simulations[sim]["players"][player_id]["_id"] = str(simulations[sim]["players"][player_id]["_id"])
                simulations[sim]["players"][player_id]["serverTime"] = simulations[sim]["players"][player_id][
                    "serverTime"].strftime('%Y-%m-%d %H:%M:%S.%f')
                simulations[sim]["players"][player_id]["roleLogIn"] = simulations[sim]["players"][player_id][
                    "roleLogIn"].strftime('%Y-%m-%d %H:%M:%S.%f')
                for collection_i in list_collections:
                    for res_i in simulations[sim]["players"][player_id][collection_i]:
                        if "serverTime" in res_i:
                            res_i["serverTime"] = res_i["serverTime"].strftime('%Y-%m-%d %H:%M:%S.%f')
                        else:
                            res_i["serverTime"] = copy.deepcopy(res_i["server-timestamp"])
                            # res_i.pop(res_i["server-timestamp"])
                        res_i["_id"] = str(res_i["_id"])
        return simulations

    def get_all_simulations_with_timeline(self, start_time, end_time):
        db_logIn = self.db["LogIn"]
        results = db_logIn.find({"sceneName": EXPERIMENT_NAME})
        filtered_results = []
        for result_i in tqdm(results, desc="Get all simulations with the players", total=results.retrieved):
            player_one_time = result_i["serverTime"]

            if player_one_time < start_time or player_one_time > end_time:
                continue
            filtered_results.append(result_i)

        return filtered_results




def compute_all_distance(simulations_data, list_collections):
    for sim in simulations_data:
        for player_id in tqdm(simulations_data[sim]["players"], desc=f"Compute distances for Players in sim {sim}"):
            for collection_i in list_collections:
                distances = []
                distances2 = []
                times = []
                timer_track_add = 0
                counter = 1
                time_counter = []
                time_prev = ""
                if collection_i != "Hand":
                    read_pos = ["x", "z"]
                else:
                    read_pos = ["x", "y", "z"]
                collection_data = simulations_data[sim]["players"][player_id][collection_i]

                if collection_i != "Hand":
                    for pointer in list(range(1, len(collection_data))):
                        point1 = collection_data[pointer - 1]
                        point2 = collection_data[pointer]
                        array_list1 = []
                        array_list2 = []
                        for pos_i in read_pos:
                            array_list1.append(point1["position"][pos_i])
                            array_list2.append(point2["position"][pos_i])
                        array1 = np.array(array_list1)
                        array2 = np.array(array_list2)
                        distance_i = compute_euclidean_distance(array1, array2)
                        distances.append(float(distance_i))
                        if pointer == 1:
                            time_prev = point1["serverTime"]
                        time_now = point2["serverTime"]
                        if time_now != time_prev:
                            datetime_prev = datetime.strptime(time_prev, '%Y-%m-%d %H:%M:%S.%f')
                            datetime_now = datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S.%f')
                            time_dif = (datetime_now - datetime_prev).total_seconds() / counter
                            time_add = copy.deepcopy(time_dif)
                            for i in range(len(distances) - len(times)):
                                time_i = (datetime_prev + timedelta(seconds=time_add)).strftime('%Y-%m-%d %H:%M:%S.%f')
                                times.append(time_i)
                                time_add += time_dif
                        time_prev = time_now
                        timer_track_add += 33
                        time_counter.append(counter)
                        counter += 1

                    time_dif = 33
                    time_add = 33
                    datetime_now = datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S.%f')
                    for i in range(len(distances) - len(times)):
                        time_i = (datetime_now + timedelta(milliseconds=time_add)).strftime('%Y-%m-%d %H:%M:%S.%f')
                        times.append(time_i)
                        time_add += time_dif
                    simulations_data[sim]["players"][player_id][f"{collection_i}_distance"] = {
                        "distances": distances,
                        "times": times
                    }
                else:
                    for pointer in list(range(2, len(collection_data), 2)):
                        point1 = collection_data[pointer - 2]
                        point2 = collection_data[pointer]
                        array_list1 = []
                        array_list2 = []
                        for pos_i in read_pos:
                            array_list1.append(point1["position"][pos_i])
                            array_list2.append(point2["position"][pos_i])
                        array1 = np.array(array_list1)
                        array2 = np.array(array_list2)
                        distance_i = compute_euclidean_distance(array1, array2)
                        distances.append(float(distance_i))
                        if pointer == 2:
                            time_prev = point1["serverTime"]
                        time_now = point2["serverTime"]
                        if time_now != time_prev:
                            datetime_prev = datetime.strptime(time_prev, '%Y-%m-%d %H:%M:%S.%f')
                            datetime_now = datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S.%f')
                            time_dif = (datetime_now - datetime_prev).total_seconds() / counter
                            time_add = 33
                            for i in range(len(distances) - len(times)):
                                time_i = (datetime_prev + timedelta(milliseconds=time_add)).strftime(
                                    '%Y-%m-%d %H:%M:%S.%f')
                                times.append(time_i)
                                time_add += time_dif

                        time_prev = time_now
                        timer_track_add += 33
                        time_counter.append(counter)
                        counter += 1

                    time_dif = 33
                    time_add = 33
                    datetime_now = datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S.%f')
                    for i in range(len(distances) - len(times)):
                        time_i = (datetime_now + timedelta(milliseconds=time_add)).strftime('%Y-%m-%d %H:%M:%S.%f')
                        times.append(time_i)
                        time_add += time_dif
                    simulations_data[sim]["players"][player_id][f"Left{collection_i}_distance"] = {
                        "distances": distances,
                        "times": times
                    }
                    for pointer in list(range(3, len(collection_data), 2)):
                        point1 = collection_data[pointer - 2]
                        point2 = collection_data[pointer]
                        array_list1 = []
                        array_list2 = []
                        for pos_i in read_pos:
                            array_list1.append(point1["position"][pos_i])
                            array_list2.append(point2["position"][pos_i])
                        array1 = np.array(array_list1)
                        array2 = np.array(array_list2)
                        distance_i = compute_euclidean_distance(array1, array2)
                        distances2.append(float(distance_i))
                        # times.append(point2["serverTime"])
                    simulations_data[sim]["players"][player_id][f"Right{collection_i}_distance"] = {
                        "distances": distances2,
                        "times": times
                    }
    return simulations_data


class Whisper2Text:
    def __init__(self, model_art: str):
        print(f"Loading Whisper Model: {model_art}")
        if torch.cuda.is_available():
            self.model = whisper.load_model(model_art, "cuda")
        else:
            self.model = whisper.load_model(model_art)

    def get_text(self, audio_file: str, remove_audio_file: bool = False):

        result = self.model.transcribe(audio_file, fp16=False, language="German")

        if remove_audio_file:
             if os.path.exists(audio_file):
                os.remove(audio_file)
        return result



def audio_saving_and_transcribe(simulations):
    whisper_data = Whisper2Text("large-v3")
    for c, sim in enumerate(simulations):
        for player_id in simulations[sim]["players"]:
            role = simulations[sim]["players"][player_id]["role"]

            audio = np.frombuffer(b'', dtype=np.uint32)
            for audio_i in simulations[sim]["players"][player_id]["Audio"]:
                audio_binary = np.frombuffer(base64.b64decode(audio_i["audio"]), dtype=np.int64)
                audio = np.concatenate((audio, audio_binary), axis=None)

            audio = pydub.AudioSegment(
                audio.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1
            )

            audio.export(f"{DATA_PATH}/audio_export_wav/{sim}_{player_id}_{role}_audio.wav", format="wav")

            transcription = whisper_data.get_text(f"{DATA_PATH}/audio_export_wav/{sim}_{player_id}_{role}_audio.wav")

            save_json(transcription, f'{DATA_PATH}/audio_export_json/{sim}_{player_id}_{role}_whisper-large-v3_transcript.json')



def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)



def merge_data_audio(data1, data_audio):
    for sim in tqdm(data1):
        for player_id in data1[sim]["players"]:
            if "Audio_transcription" in data_audio[sim]["players"][player_id]:
                data1[sim]["players"][player_id]["Audio_transcription"] = data_audio[sim]["players"][player_id]["Audio_transcription"]
            else:
                data1[sim]["players"][player_id]["Audio_transcription"] = {}
    return data1


if __name__ == '__main__':
    # Export and prepare experiment data
    list_participant = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    # Questionary Data
    questionary_results = "data/results_questionary.csv"
    pandas_read_questionary_csv(questionary_results, f"data/values_questionary.csv",
                    "data/variables_questionary.csv", list_participant)

    # Trackind Data
    start_time_experiment = datetime(1999, 1, 1, 0, 0, 0)
    end_time_experiment = datetime(2000, 1, 1, 0, 0, 0)

    db_connection = PyMongoConnect("data/config.json")
    all_sims = db_connection.get_all_simulations_with_timeline(start_time_experiment, end_time_experiment)

    all_players_timeline = db_connection.get_role_player(all_sims)

    all_data = ["Audio", "Body", "Eye", "Facial", "Hand", "Head", "Object", "Special"]
    all_infos_player = db_connection.get_all_specific_data_players(all_players_timeline, all_data)

    all_infos_player = db_connection.format_datetimes(all_infos_player, all_data)

    # Preprocess Data
    body_infos = ["Body", "Hand", "Head"]
    all_infos_players = compute_all_distance(all_infos_player, body_infos)

    save_json(all_infos_players, f"{DATA_PATH}/exported_experiments_merged_distance.json.gz", gzip_save=True)

    # Export Audio and Transcribe
    audio_saving_and_transcribe(all_infos_players)
