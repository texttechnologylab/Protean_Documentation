import datetime
import os
import os.path
import itertools


import matplotlib.pyplot as plt

from help_function import read_json, save_json
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from scipy.stats import kruskal


DATA_PATH = "exp_data/export"


def prepare_data(steps: int):
    simulation = read_json(f"{DATA_PATH}/exported_experiments_merged_distance.json.gz", True)

    body_info = {
        "Simulation": [],
        "Person": [],
        "restriction": [],
        "Distances": [],
        "Part": [],
        "times": [],
    }

    for sim in simulation:
        role_login = ""
        for player_id in tqdm(simulation[sim]["players"], desc="Get last Player LogIn"):
            if role_login == "":
                role_login = datetime.datetime.strptime(simulation[sim]["players"][player_id]["roleLogIn"],
                                                        '%Y-%m-%d %H:%M:%S.%f')
            else:
                player_roleLogIn = datetime.datetime.strptime(simulation[sim]["players"][player_id]["roleLogIn"],
                                                              '%Y-%m-%d %H:%M:%S.%f')
                if role_login < player_roleLogIn:
                    role_login = player_roleLogIn
        for player_id in tqdm(simulation[sim]["players"], desc="Get all Body Infos"):
            role_login_in = role_login
            person = simulation[sim]["players"][player_id]["role"]
            if sim not in sim_without_restrictions:
                restriction = role_to_restriction[person]
            else:
                restriction = "No Restriction"

            time_last = datetime.datetime.strptime(simulation[sim]["players"][player_id][f"Body_distance"]["times"][-1],
                                                   '%Y-%m-%d %H:%M:%S.%f')
            point_start = 0
            last_point = len(simulation[sim]["players"][player_id][f"Body_distance"]["times"])
            time_point = 0
            while role_login_in < time_last:
                while True:
                    time_i = datetime.datetime.strptime(
                        simulation[sim]["players"][player_id][f"Body_distance"]["times"][point_start],
                        '%Y-%m-%d %H:%M:%S.%f')
                    if time_i >= role_login_in:
                        break
                    else:
                        point_start += 1
                        if point_start > last_point:
                            break

                if point_start < last_point:
                    point_break = last_point
                    for c_i in range(point_start, last_point):
                        time_ci = datetime.datetime.strptime(
                            simulation[sim]["players"][player_id][f"Body_distance"]["times"][c_i],
                            '%Y-%m-%d %H:%M:%S.%f')
                        if (time_ci - role_login_in).total_seconds() / 60 > steps:
                            point_break = c_i
                            break

                    for j in ["Body", "Head", "LeftHand", "RightHand"]:
                        sum_i = sum(simulation[sim]["players"][player_id][f"{j}_distance"]["distances"][
                                    point_start:point_break])
                        body_info["Simulation"].append(sim)
                        body_info["Person"].append(person)
                        body_info["restriction"].append(restriction)
                        body_info["Distances"].append(sum_i)
                        body_info["Part"].append(j)
                        body_info["times"].append(f"{time_point}-{time_point + steps}")
                    if time_point + steps > 20:
                        break
                    time_point += 1
                    role_login_in = role_login_in + datetime.timedelta(seconds=60)
                else:
                    break
    return body_info


def body_audio_distance_visualisation():
    steps = 5
    body_info = prepare_data(steps)

    df = pd.DataFrame(body_info)
    df = df[df["Part"] != "Head"]
    df.to_csv(f"{DATA_PATH}/movement_data_steps_{steps}.csv", index=False)

    ORDER_RENAME_DICT = {"No Restriction": "No Restriction", "Vision": "Vision Restriction", "Audio": "Audio Restriction", "Interaction": "Interaction Restriction"}

    def sorter(column):
        reorder = list(ORDER_RENAME_DICT.keys())
        reorder += ["0-5", "1-6", "2-7", "3-8", "4-9", "5-10", "6-11", "7-12", "8-13", "9-14", "10-15", "11-16", "12-17", "13-18", "14-19", "15-20", "16-21"]
        print(reorder)
        cat = pd.Categorical(column, categories=reorder, ordered=True)
        return pd.Series(cat)

    df = df.sort_values(by=["restriction", "times"], key=sorter)

    df["restriction"] = df["restriction"].apply(
        lambda x: ORDER_RENAME_DICT[x] if x in ORDER_RENAME_DICT.keys() else x)


    for part in ["Body", "LeftHand", "RightHand"]:
        plt.figure(figsize=(15, 15))
        df_part = df[df["Part"] == part]
        df_part = df_part.sort_values(by=["restriction", "times"], key=sorter)

        plot = sns.lineplot(data=df_part, x="times", y="Distances", hue="restriction",
                            style="restriction")
        plot.tick_params(axis='x', rotation=90)

        plot.set_ylabel('Distance in m', size=14)
        plot.set_xlabel('Time-windows in min.', size=14)
        plot.set_yticklabels(plot.get_yticks(), size=14)
        xlabels = df_part["times"].unique()
        plot.set_xticklabels(xlabels, size=14)
        plt.setp(plot.get_legend().get_title(), fontsize='14')
        plt.setp(plot.get_legend().get_texts(), fontsize='14')
        plot.legend_.set_title(None)

        plt.tight_layout()

        plt.savefig(f"images/lineplot_{part}_{steps}_merged.png", dpi=300, transparent=True)
        plt.show()



def calculate_significance():
    steps = 5
    df = pd.read_csv(f"{DATA_PATH}/movement_data_steps_{steps}.csv")
    #df = df[df["Part"] == "Body"]
    df = df[df["Part"] == "RightHand"]
    ORDER_RENAME_DICT = {"No Restriction": "No Restriction", "Vision": "Vision Restriction", "Audio": "Audio Restriction", "Interaction": "Interaction Restriction"}

    def sorter(column):
        reorder = list(ORDER_RENAME_DICT.keys())
        reorder += ["0-5", "1-6", "2-7", "3-8", "4-9", "5-10", "6-11", "7-12", "8-13", "9-14", "10-15", "11-16", "12-17", "13-18", "14-19", "15-20", "16-21"]
        cat = pd.Categorical(column, categories=reorder, ordered=True)
        return pd.Series(cat)

    df = df.sort_values(by=["restriction", "times"], key=sorter)

    df["restriction"] = df["restriction"].apply(
        lambda x: ORDER_RENAME_DICT[x] if x in ORDER_RENAME_DICT.keys() else x)

    df_no = df[df["restriction"] == "No Restriction"]
    df_vision = df[df["restriction"] == "Vision Restriction"]
    df_audio = df[df["restriction"] == "Audio Restriction"]
    df_interaction = df[df["restriction"] == "Interaction Restriction"]

    no_distances = df_no["Distances"].values
    vision_distances = df_vision["Distances"].values
    audio_distances = df_audio["Distances"].values
    interaction_distances = df_interaction["Distances"].values

    _, p_value_no_vison = kruskal(no_distances, vision_distances)
    _, p_value_no_audio = kruskal(no_distances, audio_distances)
    _, p_value_no_interaction = kruskal(no_distances, interaction_distances)
    _, p_value_vision_audio = kruskal(vision_distances, audio_distances)
    _, p_value_vision_interaction = kruskal(vision_distances, interaction_distances)
    _, p_value_audio_interaction = kruskal(audio_distances, interaction_distances)

    print(f"No - Vision: {p_value_no_vison:f}")
    print(f"No - Audio: {p_value_no_audio:f}")
    print(f"No - Interaction: {p_value_no_interaction:f}")
    print(f"Vision - Audio: {p_value_vision_audio:f}")
    print(f"Vision - Interaction: {p_value_vision_interaction:f}")
    print(f"Audio - Interaction: {p_value_audio_interaction:f}")


sim_without_restrictions = ["1", "5", "8", "13", "14"]
role_to_restriction = {"Person 1": "Vision", "Person 2": "Audio", "Person 3": "Interaction"}

if __name__ == '__main__':

    body_audio_distance_visualisation()

    calculate_significance()
