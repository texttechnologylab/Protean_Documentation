import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama

from help_function import flat_list, argmin, read_json

DATA_PATH = "exp_data/export"

MODELS = [
    "gemma-2-9b-it-Q5_K_M.gguf",
    "gemma-2-27b-it-Q5_K_M.gguf",
    "leo-hessianai-13b-chat.Q5_K_M.gguf",
    "llama-2-13b-german-assistant-v4.Q5_K_M.gguf",
    "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
    "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
    "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
    "tiiuae-falcon-7b-instruct-Q5_K_M.gguf",
    "c4ai-command-r-08-2024-Q5_K_M.gguf",
    "Crimson_Dawn-v0.2_Q5_K_M.gguf",
    "em_german_13b_leo_Q5_K_M.gguf",
    "Llama-3.1-Storm-8B.Q5_K_M.gguf",
]

EVAL_SCORES = [["BS02_01", "BS02_02", "BS02_03", "BS02_04", "BS02_05", "BS02_06"],
               ["BS03_01", "BS03_02", "BS03_03", "BS03_04"],
               ["BI01_01", "BI01_02", "BI01_03", "BI01_04", "BI01_05"],
               ["BI02_01", "BI02_02", "BI02_03", "BI02_04", "BI02_05"],
               ["WB01_01", "WB01_02", "WB01_03", "WB01_04", "WB01_05"],
               ["SZ01_01", "SZ01_04", "SZ01_08", "SZ01_09", "SZ01_10", "SZ01_11"],
               ["SZ05_08", "SZ05_05", "SZ05_07", "SZ05_09"],
               ["SZ02_01", "SZ02_02", "SZ02_03", "SZ02_04"]]

EXPERIMENT_REST = [False, True, True, True, False, True, True, False, True, True, True, True, False, False, True]

FREE_TEXT_LABELS = {"BS04_01": "Controls", "BS05_01": "Interfaces", "BI03_01": "User Interaction", "BI04_01": "Avatar",
                    "WB02_01": "Wellbeing", "SZ04_01": "Tutorial", "SZ06_01": "VR Experience",
                    "SZ03_01": "Szenario"}

INV_LIST = ["BS01_02", "BS01_04",
            "BS02_03", "BS02_05", "BS02_06",
            "BS03_02", "BS03_04",
            "BI01_03",
            "BI02_02", "BI02_03", "BI02_04", "BI02_05",
            "WB01_03", "WB01_04", "WB01_05",
            "SZ02_01", "SZ02_03",
            "SZ05_05", "SZ05_07",
            "SZ01_03", "SZ01_09", "SZ01_010", "SZ01_11",
            ]

SENTIMENT_TAGGER = ["sentiment_huggingface_cardiffnlp_twitter-xlm-roberta-base-sentiment",
                    "sentiment_huggingface_LiYuan_amazon-review-sentiment-analysis",
                    "sentiment_huggingface_mdraw_german-news-sentiment-bert",
                    "sentiment_huggingface_nlptown_bert-base-multilingual-uncased-sentiment",
                    "sentiment_huggingface_oliverguhr_german-sentiment-bert",
                    "sentiment_vader",
                    "sentiment_textblob"
                    ]


EVAL_SCORES_FLAT = flat_list(EVAL_SCORES)


class ExperimentDataManager:
    def __init__(self, folder_path):
        self.folder_path = folder_path

        self.experiment_list = []
        self.role_list = []
        self.transcription_list = []

        self._load_all_experiment(False)

    def __len__(self):
        return max(self.experiment_list)

    def _load_all_experiment(self, normalized=False):
        for filename in os.listdir(self.folder_path):
            f = os.path.join(self.folder_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                if (not normalized and f.endswith("audio.wav_large_transcript.json")) \
                        or (normalized and f.endswith("audio_normalized.wav_large_transcript.json")):
                    experiment, player_id, role, _ = filename.split("_", 3)
                    role = int(role[-1])
                    experiment = int(experiment)

                    data = read_json(f)
                    self.experiment_list.append(experiment)
                    self.role_list.append(role)
                    self.transcription_list.append(data)


    def get_all_transcription(self, experiment):
        result = [{"segments": []}, {"segments": []}, {"segments": []}]
        for i in range(len(self.experiment_list)):
            if self.experiment_list[i] == experiment:
                result[self.role_list[i]-1] = self.transcription_list[i]
        return result

    def get_metadata_for_person(self, experiment, person):
        f = os.path.join("meta_data", f"{experiment}_Person {person}_metadata.json")
        if not os.path.isfile(f):
            return {"Audio": {"localTime": 9999999999}}
        data = read_json(f)
        return data


    def get_audio_start_for_person(self, experiment, person):
        meta_data = self.get_metadata_for_person(experiment, person)
        return meta_data["Audio"]["localTime"]

    def get_turns_for_experiment(self, experiment):
        transcriptions = self.get_all_transcription(experiment)
        transcriptions[0]["segments"].append({"start": 9999999999, "end": 9999999999})
        transcriptions[1]["segments"].append({"start": 9999999999, "end": 9999999999})
        transcriptions[2]["segments"].append({"start": 9999999999, "end": 9999999999})

        audio_starts = [self.get_audio_start_for_person(experiment, i) for i in range(1, 4)]
        audio_starts = [x - min(audio_starts) for x in audio_starts]

        pointer_player = [0, 0, 0]

        turn_list = []
        while (pointer_player[0] < len(transcriptions[0]["segments"]) or
               pointer_player[1] < len(transcriptions[1]["segments"]) or
               pointer_player[2] < len(transcriptions[2]["segments"])):


            transcription_timing = [transcriptions[0]["segments"][pointer_player[0]]["start"] + audio_starts[0],
                                    transcriptions[1]["segments"][pointer_player[1]]["start"] + audio_starts[1],
                                    transcriptions[2]["segments"][pointer_player[2]]["start"] + audio_starts[2]]

            next_turn = argmin(transcription_timing)

            if transcription_timing[next_turn] > 9999999998:
                break
            if transcriptions[next_turn]["segments"][pointer_player[next_turn]]["text"] != "":
                turn_list.append({"player": next_turn + 1, "start": transcription_timing[next_turn], "text": transcriptions[next_turn]["segments"][pointer_player[next_turn]]["text"]})
                #print(f"Player {next_turn + 1}; Start {transcription_timing[next_turn]:.2f}:", transcriptions[next_turn]["segments"][pointer_player[next_turn]]["text"])
            pointer_player[next_turn] += 1

        return turn_list


    def get_turns_on_words_for_experiment(self, experiment: list):
        transcriptions_all = self.get_all_transcription(experiment)

        word_list_1 = [x["words"] for x in transcriptions_all[0]["segments"]]
        #print("2")
        word_list_2 = [x["words"] for x in transcriptions_all[1]["segments"]]
        #print("3")
        word_list_3 = [x["words"] for x in transcriptions_all[2]["segments"]]

        word_lists = [flat_list(word_list_1), flat_list(word_list_2), flat_list(word_list_3)]

        word_lists[0].append({"word": ["[END]"], "start": 9999999999, "end": 9999999999})
        word_lists[1].append({"word": ["[END]"], "start": 9999999999, "end": 9999999999})
        word_lists[2].append({"word": ["[END]"], "start": 9999999999, "end": 9999999999})

        audio_starts = [self.get_audio_start_for_person(experiment, i) for i in range(1, 4)]
        audio_starts = [x - min(audio_starts) for x in audio_starts]

        pointer_player = [0, 0, 0]

        turn_list = []
        while (pointer_player[0] < len(word_lists[0]) or
               pointer_player[1] < len(word_lists[1]) or
               pointer_player[2] < len(word_lists[2])):


            transcription_timing = [word_lists[0][pointer_player[0]]["start"] + audio_starts[0],
                                    word_lists[1][pointer_player[1]]["start"] + audio_starts[1],
                                    word_lists[2][pointer_player[2]]["start"] + audio_starts[2]]

            transcription_timing_end = [word_lists[0][pointer_player[0]]["start"] + audio_starts[0],
                                    word_lists[1][pointer_player[1]]["start"] + audio_starts[1],
                                    word_lists[2][pointer_player[2]]["start"] + audio_starts[2]]

            next_turn = argmin(transcription_timing)

            if transcription_timing[next_turn] > 9999998:
                break
            if word_lists[next_turn][pointer_player[next_turn]]["word"] != "":
                turn_list.append({"player": next_turn + 1, "start": transcription_timing[next_turn], "text": word_lists[next_turn][pointer_player[next_turn]]["word"]})
                #print(f"Player {next_turn + 1}; Start {transcription_timing[next_turn]:.2f}:", transcriptions[next_turn]["segments"][pointer_player[next_turn]]["text"])
            pointer_player[next_turn] += 1

        # merge turns with same player
        new_turn_list = []
        for turn in turn_list:
            if len(new_turn_list) == 0 or new_turn_list[-1]["player"] != turn["player"]:
                new_turn_list.append(turn)
            else:
                new_turn_list[-1]["text"] += turn["text"]

        return new_turn_list


def create_df_from_json(json_merged, invert=False):
    df_data = []
    for experiment in json_merged.keys():
        for person in json_merged[experiment].keys():
            person_data = json_merged[experiment][person]
            person_dict = {"experiment": int(experiment), "person_id": person, "person_role": person_data["DI01"],
                           "restriction": -1 if not EXPERIMENT_REST[int(experiment) - 1] else person_data["DI01"],
                           "feeled_restriction": person_data["DI02"] - 2 if person_data["DI02"] == 1 else person_data[
                                                                                                              "DI02"] - 1,
                           "started": person_data["STARTED"]}

            # print(person_data["DI01"])
            for score in EVAL_SCORES_FLAT:
                if invert and score in INV_LIST and person_data[score] is not None and person_data[score] >= 0:
                    person_dict[score] = 6 - person_data[score]  # skale is from 1-5 ...
                else:
                    person_dict[score] = person_data[score]

            for free_text in FREE_TEXT_LABELS.keys():
                sentiment = person_data[free_text]
                # print(sentiment)
                if sentiment["text"] is None or len(sentiment["text"]) < 10:
                    continue
                person_dict[f"senttext:{free_text}"] = sentiment["text"]
                # print(sentiment)
                sentiment_sum = 0
                sentiment_count = 0
                for tagger_id, sent_tagger in enumerate(SENTIMENT_TAGGER):
                    # print(sent_tagger)

                    if sentiment["text"] is None:
                        person_dict[f"sent:{free_text}:{tagger_id}"] = None
                    else:
                        person_dict[f"sent:{free_text}:{tagger_id}"] = sentiment[sent_tagger]["avg_text"]
                        # person_dict[f"sent:{free_text}:{tagger_id}"] = sentiment[sent_tagger]["full_text"]

                        sentiment_sum += sentiment[sent_tagger]["avg_text"]
                        # sentiment_sum += sentiment[sent_tagger]["full_text"]
                        sentiment_count += 1
                person_dict[f"sentavg:{free_text}"] = sentiment_sum / sentiment_count if sentiment_count > 0 else None
            df_data.append(person_dict)
    df = pd.DataFrame(df_data)
    return df


def load_model(model_name: str, emb: bool, gpu=-1) -> Llama:
    llm = Llama(
        model_path=f"{DATA_PATH}/models/gguf/{model_name}",
        n_gpu_layers=gpu,  # Uncomment to use GPU acceleration
        verbose=False,
        n_ctx=1024,
        embedding=emb
    )

    return llm


def questionary_prompt(df, llm, prompt_template, model_name, output_file):
    if os.path.exists(f"{output_file}_{model_name}.csv"):
        return

    results = []
    input_text = []
    for row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {output_file} {model_name}"):
        text = str(row[1]["value"])
        text = text.replace("\n", " ")
        input_text.append(text)
        # Replace {text} in the provided prompt
        formatted_prompt = prompt_template.format(text=text)

        output = llm.create_chat_completion(
            temperature=0,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
        )

        results.append(output["choices"][0]["message"]["content"])

    df["input_text"] = input_text
    df["sentiment"] = results
    df.to_csv(f"{DATA_PATH}/questionary_prompt/{output_file}_{model_name}.csv", encoding='utf-8')


def generate_and_save_questionary_embeddings(df, llm, model_name, output_dir):
    if not os.path.isdir(f"{DATA_PATH}/{output_dir}_{model_name}"):
        os.mkdir(f"{DATA_PATH}/{output_dir}_{model_name}")

    for row in tqdm(df.iterrows(), total=len(df), desc=f"Processing Questionary Emb {output_dir} {model_name}"):
        text = str(row[1]["value"])
        exp = row[1]["experiment"]
        person_id = row[1]["person_id"]
        text_field = row[1]["variable"]
        text_field = text_field.split(":")[1]

        if os.path.exists(f"{DATA_PATH}/{output_dir}_{model_name}/{exp}_{person_id}_{text_field}_emb.npy"):
            continue

        text = text.replace("\n", " ")

        embeddings = llm.create_embedding(text)
        embeddings = embeddings["data"][0]["embedding"]
        emb = np.array(embeddings)
        np.save(f"{DATA_PATH}/{output_dir}_{model_name}/{exp}_{person_id}_{text_field}_emb.npy", emb)


def generate_and_save_transcription_embeddings(exp_data, llm, model_name):
    if not os.path.isdir(f"{DATA_PATH}/transcription_embeddings_{model_name}"):
        os.mkdir(f"{DATA_PATH}/transcription_embeddings_{model_name}")

    experiment_list = exp_data.experiment_list
    role_list = exp_data.role_list
    transcription_list = exp_data.transcription_list
    for exp_id, role, transcription in tqdm(zip(experiment_list, role_list, transcription_list),
                                            total=len(experiment_list),
                                            desc=f"Processing Transcription Emb {model_name}"):
        if os.path.exists(f"{DATA_PATH}/transcription_embeddings_{model_name}/{exp_id}_{role}_emb.npy"):
            continue

        embeddings = llm.create_embedding(transcription["text"])
        embeddings = embeddings["data"][0]["embedding"]
        emb = np.array(embeddings)
        np.save(f"{DATA_PATH}/transcription_embeddings_{model_name}/{exp_id}_{role}_emb.npy", emb)


def generate_and_save_turn_embeddings(turn_data, llm, model_name, wordbase=False):
    if not os.path.isdir(f"{DATA_PATH}/turn_embeddings_wordbase_{wordbase}_{model_name}"):
        os.mkdir(f"{DATA_PATH}/turn_embeddings_wordbase_{wordbase}_{model_name}")

    for turn in tqdm(turn_data, desc=f"Processing Turn Emb {model_name}"):
        exp_id = turn["experiment"]
        turn_id = turn["turn"]
        player_x = turn["turns"][0]["player"]
        player_y = turn["turns"][1]["player"]
        if os.path.exists(
                f"{DATA_PATH}/turn_embeddings_wordbase_{wordbase}_{model_name}/{exp_id}_{turn_id}_{player_x}_{player_y}_emb.npy"):
            continue

        text = f"Person A: '{turn['turns'][0]['text']}'; Person B: '{turn['turns'][1]['text']}'"
        embeddings = llm.create_embedding(text)
        embeddings = embeddings["data"][0]["embedding"]
        emb = np.array(embeddings)
        np.save(
            f"{DATA_PATH}/turn_embeddings_wordbase_{wordbase}_{model_name}/{exp_id}_{turn_id}_{player_x}_{player_y}_emb.npy",
            emb)


def create_turn_groups_of_2(exp_data):
    grp_data = []
    for exp_id in range(1, len(exp_data) + 1):
        turn_data = exp_data.get_turns_for_experiment(exp_id)
        for turn_id in range(len(turn_data) - 1):
            if turn_data[turn_id]['player'] != turn_data[turn_id + 1]['player']:
                if len(turn_data[turn_id]["text"]) > 3 and len(turn_data[turn_id + 1]["text"]) > 3:
                    grp_data.append({"experiment": exp_id, "turn": turn_id + 1,
                                     "turns": [turn_data[turn_id], turn_data[turn_id + 1]]})

    return grp_data


def create_turn_groups_of_2_wordbase(exp_data):
    group_data = []
    for exp_id in range(1, len(exp_data) + 1):
        turn_data = exp_data.get_turns_on_words_for_experiment(exp_id)
        for turn_id in range(len(turn_data) - 1):
            if (turn_data[turn_id]['player'] != turn_data[turn_id + 1]['player']):
                if (len(turn_data[turn_id]["text"]) > 3 and len(turn_data[turn_id + 1]["text"]) > 3):
                    group_data.append({"experiment": exp_id, "turn": turn_id + 1,
                                       "turns": [turn_data[turn_id], turn_data[turn_id + 1]]})

    return group_data


def turn_prompt(turn_data, llm, prompt_template, model_name, output_file):
    if os.path.exists(f"{output_file}_{model_name}.csv"):
        return

    results = []
    for turn in tqdm(turn_data, desc=f"Processing Turn Promt {output_file} {model_name}"):
        exp_id = turn["experiment"]
        turn_id = turn["turn"]
        player_x = turn["turns"][0]["player"]
        player_y = turn["turns"][1]["player"]
        input_for_promt = f"Turn: Person X says: {turn['turns'][0]['text']} \n Person Y responds to Person X: {turn['turns'][1]['text']}"
        formatted_prompt = prompt_template.format(input_for_promt=input_for_promt)

        try:
            output = llm.create_chat_completion(
                temperature=0,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
            )
            result = output["choices"][0]["message"]["content"]
        except Exception:
            result = "ERROR"

        results.append({"exp_id": exp_id, "turn_id": turn_id, "player_x": player_x, "player_y": player_y,
                        "player_x_start": turn['turns'][0]['start'], "player_y_start": turn['turns'][1]['start'],
                        "player_x_text": turn['turns'][0]['text'], "player_y_text": turn['turns'][1]['text'],
                        "sentiment": result})

    df = pd.DataFrame(results)
    df.to_csv(f"{DATA_PATH}/turn_prompt/{output_file}_{model_name}.csv", encoding='utf-8')


if __name__ == "__main__":

    # Questionary Data
    json_merge = read_json(f"{DATA_PATH}/all_simulation_evaluation_infos_sentiment.json")

    df_inv = create_df_from_json(json_merge, invert=True)

    text_colums = [col for col in df_inv if col.startswith('senttext:')]
    eval_df = df_inv[['experiment', 'person_id'] + text_colums]

    eval_df = eval_df.melt(id_vars=['experiment', 'person_id'], value_vars=text_colums)

    experiment_data = ExperimentDataManager(f"{DATA_PATH}/audio_export_json")
    group_data = create_turn_groups_of_2(experiment_data)

    # Run LLM for all models
    for model in tqdm(MODELS):
        ll_model = load_model(model, emb=False)


        generate_and_save_questionary_embeddings(eval_df, ll_model, model, "questionary_embeddings")
        generate_and_save_transcription_embeddings(experiment_data, ll_model, model)
        generate_and_save_turn_embeddings(group_data, ll_model, model, wordbase=False)

        #############################
        prompt = """You are an expert in German linguistics. Carry out a sentiment analysis of the given statement and assign a score between -1 and +1. A score of +1 means that the statement is exceptionally positive, while -1 means that the statement is exceptionally negative.

        Statement: {text}

        Sentiment score: your answer;

        To make sure we have found the right answer, we should go through this step by step. Only output the sentiment score and nothing else!
        """
        questionary_prompt(eval_df, ll_model, prompt, model, "questionary_sentiment_enpromt")

        #############################

        prompt = """
        Sie sind ein Experte für deutsche Linguistik. Führen Sie eine Sentimentanalyse der gegebenen Aussage durch und weisen Sie einen Score zwischen -1 und +1 zu. Ein Score von +1 bedeutet, dass die Aussage außergewöhnlich positiv ist, während -1 bedeutet, dass die Aussage außergewöhnlich negativ ist.

        Aussage: {text}

        Sentimentscore: deine Antwort;

        Um sicher zu gehen, dass wir die richtige Antwort gefunden haben, sollten wir dies Schritt für Schritt durchspielen. Geben Sie nur den Sentimentscore aus und sonst nichts!
        """
        questionary_prompt(eval_df, ll_model, prompt, model, "questionary_sentiment_depromt")

        #############################

        prompt = """
        You are an expert in German linguistics. Analyse the objectivity/subjectivity of the given statement and assign a score between 0 and 1. A score of 1 means that the statement is exceptionally subjective, while 0 means that the statement is completely objective.

        Statement: {text}

        Objectivity/Subjectivity Score: your answer;

        To make sure we have found the right answer, we should go through this step by step. Output only the score and nothing else!
        """
        questionary_prompt(eval_df, ll_model, prompt, model, "questionary_subjectivity_enpromt")

        #############################

        prompt = """
        Sie sind ein Experte für deutsche Linguistik. Führen Sie eine Analyse der Objektivität/Subjektivität der gegebenen Aussage durch und weisen Sie einen Score zwischen 0 und 1 zu. Ein Score von 1 bedeutet, dass die Aussage außergewöhnlich subjektiv ist, während 0 bedeutet, dass die Aussage vollkommen objektiv ist.

        Aussage: {text}

        Objektivität/Subjektivität Score: deine Antwort;

        Um sicher zu gehen, dass wir die richtige Antwort gefunden haben, sollten wir dies Schritt für Schritt durchspielen. Geben Sie nur den Score aus und sonst nichts!
        """
        questionary_prompt(eval_df, ll_model, prompt, model, "questionary_subjectivity_depromt")

        #############################

        prompt = """
        You are an expert in German linguistics. Analyse the emotionality of the given statement and assign a score between 0 and 1. A score of 1 means that the statement is exceptionally emotional, while 0 means that the statement contains no emotional colouring at all.

        Statement: {text}

        Emotionality score: your answer;

        To make sure we have found the right answer, we should go through this step by step. Output only the score and nothing else!
        """
        questionary_prompt(eval_df, ll_model, prompt, model, "questionary_emotionality_enpromt")

        #############################

        prompt = """
        Sie sind ein Experte für deutsche Linguistik. Führen Sie eine Analyse der Emotionalität der gegebenen Aussage durch und weisen Sie einen Score zwischen 0 und 1 zu. Ein Score von 1 bedeutet, dass die Aussage außergewöhnlich emotional ist, während 0 bedeutet, dass die Aussage keinerlei emotionale Färbung enthält.

        Aussage: {text}

        Emotionalitäts-Score: deine Antwort;

        Um sicher zu gehen, dass wir die richtige Antwort gefunden haben, sollten wir dies Schritt für Schritt durchspielen. Geben Sie nur den Score aus und sonst nichts!
        """
        questionary_prompt(eval_df, ll_model, prompt, model, "questionary_emotionality_depromt")

        ###############################

        prompt = '''
As a linguistic annotator, your task is to analyze the type of turns of a conversation following Dialogue Act Annotation.

A type is a class of dialogue acts concerned with one particular aspect of communication that a dialogue act can address independently from other types. In order to classify turns, consider the following set of eight types according to Dialogue Act Annotation:

1. Task (or Activity): dialogue acts dealing with the task or activity that motivates the dialogue.
2. Auto-and Allo-Feedback: dialogue acts providing or eliciting information about the processing of previous utterances by the current speaker (auto) or the current addressee (allo). Examples: "Huh?", "What?", "True", "Sure", "That's right."
3. Turn Management: activities for obtaining, keeping, releasing, or assigning the right to speak. Examples: final intonational rise, "Henry, can you..."
4. Time Management: acts for managing the use of time in the interaction. Examples: "Well...", "Just a minute", slowing down speech
5. Discourse Structuring: dialogue acts dealing with topic management, opening and closing (sub-)dialogues, or otherwise structuring the dialogue. Examples: "Question:"
6. Own and Partner Communication Management: actions by the speaker for editing his current contribution, or for editing (e.g. completing) the current contribution of another current speaker. Examples: "Oh sorry, no", "Or no wait", completion or correction of the utterance of the interlocutor
7. Social Obligations Management: dialogue acts for dealing with social conventions such as greeting, introducing oneself, apologizing, and thanking, and responses to these acts, such as accepting an apology. Example: "I'm sorry", "Hello", "Good morning", "Thanks"
8. Broken: dialogue acts that cannot be classified into any of the other dimensions or is incomplete.

Your task is to analyze the type of turns.

Follow these steps to complete your analysis:
Step 1: Analyze the types of the turn.
Step 2: For each type identified, provide a confidence value in the range from 0 (maximum uncertainty) to 1 (maximum certainty), indicating how confident you are in your analysis.
Step 3: Output your analysis according to the following scheme:

Type: your answer;
Confidence: your answer;

Analyze the following type using this method:
<input>{input_for_promt}</input>

Let’s work this out in a step-by-step way to be sure we have the right answer. Output only the latter scheme and nothing else!
    '''
        turn_prompt(group_data, ll_model, prompt, model, "transcriptions_turns_classifications_enpromt")


        prompt = '''
Als linguistischer Annotator ist es deine Aufgabe, die Art der Gesprächsbeiträge anhand der Dialogakt-Annotierung zu analysieren.

Ein Typ ist eine Klasse von Dialogakten, die sich mit einem bestimmten Aspekt der Kommunikation befasst, den ein Dialogakt unabhängig von anderen Typen ansprechen kann. Um Gesprächsbeiträge zu klassifizieren, beachte bitte das folgende Set von acht Typen gemäß der Dialogakt-Annotierung:

1. Aufgabe (oder Aktivität): Dialogakte, die sich mit der Aufgabe oder Aktivität befassen, die den Dialog motiviert.
2. Auto- und Allo-Feedback: Dialogakte, die Informationen über die Verarbeitung früherer Äußerungen durch den aktuellen Sprecher (Auto) oder den aktuellen Adressaten (Allo) liefern oder erfragen. Beispiele: "Hä?", "Was?", "Stimmt", "Sicher", "Das ist richtig."
3. Gesprächssteuerung: Aktivitäten, um das Recht zu sprechen zu erhalten, zu behalten, abzugeben oder zuzuweisen. Beispiele: Endsteigende Intonation, "Henry, kannst du..."
4. Zeitmanagement: Akte zur Verwaltung der Zeitnutzung in der Interaktion. Beispiele: "Also...", "Einen Moment", verlangsamtes Sprechen
5. Diskursstrukturierung: Dialogakte, die sich mit dem Themenmanagement, dem Eröffnen und Schließen (Teil-)Dialogen oder der sonstigen Strukturierung des Dialogs befassen. Beispiele: "Frage:"
6. Eigene und Partner-Kommunikationsverwaltung: Aktionen des Sprechers zur Bearbeitung seines aktuellen Beitrags oder zur Bearbeitung (z. B. Vervollständigung) des aktuellen Beitrags eines anderen Sprechers. Beispiele: "Oh, sorry, nein", "Oder nein, warte", Vervollständigung oder Korrektur der Äußerung des Gesprächspartners
7. Verwaltung sozialer Verpflichtungen: Dialogakte, die sich mit sozialen Konventionen wie Begrüßung, Vorstellung, Entschuldigung und Danksagung befassen, sowie Antworten auf diese Akte, wie das Annehmen einer Entschuldigung. Beispiele: "Entschuldigung", "Hallo", "Guten Morgen", "Danke"
8. Defekt: Dialogakte, die in keine der anderen Dimensionen eingeordnet werden können oder unvollständig sind.

Deine Aufgabe ist es, die Art der Gesprächsbeiträge zu analysieren.

Befolge diese Schritte, um deine Analyse durchzuführen:
Schritt 1: Analysiere die Arten des Gesprächsbeitrags.
Schritt 2: Gib für jeden identifizierten Typ einen Vertrauenswert im Bereich von 0 (maximale Unsicherheit) bis 1 (maximale Sicherheit) an, der angibt, wie sicher du dir bei deiner Analyse bist.
Schritt 3: Gib deine Analyse nach folgendem Schema aus:

Typ: deine Antwort;
Vertrauen: deine Antwort;

Analysiere den folgenden Typ nach dieser Methode:
<input>{input_for_promt}</input>

Lass uns das Schritt für Schritt durchgehen, um sicherzustellen, dass wir die richtige Antwort haben. Gib nur das oben genannte Schema aus und nichts anderes!
'''

        turn_prompt(group_data, ll_model, prompt, model, "transcriptions_turns_classifications_depromt")


        prompt = '''
As a linguistic annotator, your task is to analyze the type of conversational contributions based on dialogue act annotation.

A type is a class of dialogue acts that addresses a specific aspect of communication, which a dialogue act can address independently of other types. To classify conversational contributions, please consider the following set of eight types according to dialogue act annotation:

1. Question (or Activity): The speaker asks for information or requests clarification. Examples: "How are you?", "Can you send me the file?", "When does the meeting start?"
2. Statement/Information: The speaker shares information, opinions, or statements. This also includes self-corrections. Examples: "The weather is very nice today.", "In my opinion, that's a good idea.", "Oh, I meant 2 PM, not 1 PM."
3. Request: The speaker asks the interlocutor to perform an action or makes a suggestion. Examples: "Please close the window.", "Can you help me?", "I'll send you the file tomorrow."
4. Confirmation/Denial: The speaker signals agreement or disagreement with a previous statement or request. Examples: "Yes, that's true.", "No, I see it differently.", "Exactly, that's what I meant."
5. Social Act: The speaker performs a social action, such as greetings, thanks, apologies, or compliments. This also includes simple feedback or closing a conversation. Examples: "Hello, how are you?", "Thank you for your help.", "You did a great job.", "Sorry I'm late."
6. Defect: Dialogue acts that cannot be categorized into any of the other dimensions or are incomplete.

Your task is to analyze the type of conversational contribution.

Follow these steps to perform your analysis:
Schritt 1: Analyze the types of conversational contributions.
Schritt 2: For each identified type, provide a confidence score ranging from 0 (maximum uncertainty) to 1 (maximum certainty), indicating how confident you are in your analysis.
Schritt 3: Output your analysis using the following format:

Type: your response; 
Confidence: your response;

Analyze the following type using this method: 
<input>{input_for_promt}</input>

Let’s go through this step by step to make sure we get the correct answer. Only provide the output in the format given above and nothing else! 
'''

        turn_prompt(group_data, ll_model, prompt, model, "transcriptions_turns_alt_classifications_enpromt")


        prompt = '''
Als linguistischer Annotator ist es deine Aufgabe, die Art der Gesprächsbeiträge anhand der Dialogakt-Annotierung zu analysieren.

Ein Typ ist eine Klasse von Dialogakten, die sich mit einem bestimmten Aspekt der Kommunikation befasst, den ein Dialogakt unabhängig von anderen Typen ansprechen kann. Um Gesprächsbeiträge zu klassifizieren, beachte bitte das folgende Set von acht Typen gemäß der Dialogakt-Annotierung:

1. Frage (oder Aktivität): Der Sprecher stellt eine Informationsanfrage oder bittet um Klarstellung. Beispiele: "Wie geht es dir?", "Kannst du mir die Datei schicken?", "Wann fängt das Meeting an?"
2. Aussage/Information: Der Sprecher teilt Informationen, Meinungen oder Feststellungen mit. Dies umfasst auch Selbstkorrekturen. Beispiele: "Das Wetter ist heute sehr schön.", "Meiner Meinung nach ist das eine gute Idee.", "Oh, ich meinte 14 Uhr, nicht 13 Uhr."
3. Aufforderung : Der Sprecher fordert den Gesprächspartner auf, eine Handlung auszuführen oder unterbreitet einen Vorschlag. Beispiele: "Bitte schließ das Fenster.", "Kannst du mir helfen?", "Ich schicke dir die Datei morgen."
4. Bestätigung/Verneinung: Der Sprecher signalisiert Zustimmung oder Ablehnung einer vorherigen Aussage oder Aufforderung. Beispiele: "Ja, das stimmt.", "Nein, das sehe ich anders.", "Genau, das meinte ich."
5. Sozialer Akt: Der Sprecher vollzieht eine soziale Handlung, wie Begrüßungen, Danksagungen, Entschuldigungen oder Lob. Dies umfasst auch einfaches Feedback oder das Schließen eines Gesprächs. Beispiele: "Hallo, wie geht’s?", "Vielen Dank für deine Hilfe.", "Das hast du gut gemacht.", "Entschuldigung, dass ich zu spät bin."
6. Defekt: Dialogakte, die in keine der anderen Dimensionen eingeordnet werden können oder unvollständig sind.

Deine Aufgabe ist es, die Art der Gesprächsbeiträge zu analysieren.

Befolge diese Schritte, um deine Analyse durchzuführen:
Schritt 1: Analysiere die Arten des Gesprächsbeitrags.
Schritt 2: Gib für jeden identifizierten Typ einen Vertrauenswert im Bereich von 0 (maximale Unsicherheit) bis 1 (maximale Sicherheit) an, der angibt, wie sicher du dir bei deiner Analyse bist.
Schritt 3: Gib deine Analyse nach folgendem Schema aus:

Typ: deine Antwort;
Vertrauen: deine Antwort;

Analysiere den folgenden Typ nach dieser Methode:
<input>{input_for_promt}</input>

Lass uns das Schritt für Schritt durchgehen, um sicherzustellen, dass wir die richtige Antwort haben. Gib nur das oben genannte Schema aus und nichts anderes!
'''
        turn_prompt(group_data, ll_model, prompt, model, "transcriptions_turns_alt_classifications_depromt")



