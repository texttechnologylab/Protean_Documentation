import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pacmap
import umap

import seaborn as sns
import matplotlib.pyplot as plt

from help_function import read_json, flat_list

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
    "c4ai-command-r-08-2024-Q5_K_M.gguf",
    "Crimson_Dawn-v0.2_Q5_K_M.gguf",
    "Llama-3.1-Storm-8B.Q5_K_M.gguf",
]

EVAL_SCORES = [["BS02_01", "BS02_02", "BS02_03", "BS02_04", "BS02_05", "BS02_06"],
               ["BS03_01", "BS03_02", "BS03_03", "BS03_04"],
               # ["BS03_01", "BS03_02", "BS03_03", "BS03_04"],
               ["BI01_01", "BI01_02", "BI01_03", "BI01_04", "BI01_05"],
               ["BI02_01", "BI02_02", "BI02_03", "BI02_04", "BI02_05"],
               ["WB01_01", "WB01_02", "WB01_03", "WB01_04", "WB01_05"],
               ["SZ01_01", "SZ01_04", "SZ01_08", "SZ01_09", "SZ01_10", "SZ01_11"],
               # ["SZ01_01", "SZ01_02", "SZ01_04", "SZ01_06", "SZ01_08", "SZ01_09", "SZ01_10", "SZ01_11"],
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

SENTIMENT_TAGGER = [#"sentiment_huggingface_cardiffnlp_twitter-xlm-roberta-base-sentiment",
                    #"sentiment_huggingface_clampert_multilingual-sentiment-covid19",
                    #"sentiment_huggingface_LiYuan_amazon-review-sentiment-analysis",
                    #"sentiment_huggingface_mdraw_german-news-sentiment-bert",
                    #"sentiment_huggingface_nlptown_bert-base-multilingual-uncased-sentiment",
                    #"sentiment_huggingface_oliverguhr_german-sentiment-bert",
                    #"sentiment_huggingface_philschmid_distilbert-base-multilingual-cased-sentiment-2",
                    #"sentiment_vader",
                    #"sentiment_textblob"
                    ]

EVAL_SCORES_FLAT = flat_list(EVAL_SCORES)

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
                        sentiment_sum += sentiment[sent_tagger]["avg_text"]
                        sentiment_count += 1
                person_dict[f"sentavg:{free_text}"] = sentiment_sum / sentiment_count if sentiment_count > 0 else None
            df_data.append(person_dict)
    df = pd.DataFrame(df_data)
    return df


random_state = 123

def get_new_models():
    red_mod = []

    red_mod.append({"name": "pacmap", "model": pacmap.PaCMAP(n_components=2, random_state=random_state) })

    red_mod.append({"name": "pca_full", "model": PCA(n_components=2, svd_solver="full", random_state=random_state)})
    red_mod.append({"name": "pca_arpack", "model": PCA(n_components=2, svd_solver="arpack", random_state=random_state)})
    red_mod.append({"name": "pca_randomized", "model": PCA(n_components=2, svd_solver="randomized", random_state=random_state)})

    red_mod.append({"name": "tsne_10", "model": TSNE(n_components=2, random_state=random_state, perplexity=10)})
    red_mod.append({"name": "tsne_20", "model": TSNE(n_components=2, random_state=random_state, perplexity=20)})
    red_mod.append({"name": "tsne_40", "model": TSNE(n_components=2, random_state=random_state, perplexity=40)})
    red_mod.append({"name": "tsne_100", "model": TSNE(n_components=2, random_state=random_state, perplexity=100)})

    red_mod.append({"name": "umap_10", "model": umap.UMAP(n_components=2, random_state=random_state, n_neighbors=10) })
    red_mod.append({"name": "umap_20", "model": umap.UMAP(n_components=2, random_state=random_state, n_neighbors=20) })
    red_mod.append({"name": "umap_40", "model": umap.UMAP(n_components=2, random_state=random_state, n_neighbors=40) })
    red_mod.append({"name": "umap_100", "model": umap.UMAP(n_components=2, random_state=random_state, n_neighbors=100) })
    return red_mod


DATA_PATH = "exp_data/export"
if __name__ == "__main__":
    # Prepare Questionary Dataframe
    json_merge = read_json(f"{DATA_PATH}/all_simulation_evaluation_infos_sentiment.json")

    df_inv = create_df_from_json(json_merge, invert=True)

    # iterate over df rows
    person_id_to_restriction_map = {}
    for index, row in df_inv.iterrows():
        person_id = row["person_id"]
        restriction = row["restriction"]
        person_id_to_restriction_map[person_id] = restriction


    for model in MODELS: # transcription_embeddings
        for emb_source in ["questionary_embeddings", "transcription_embeddings" , "turn_embeddings_wordbase_False", "turn_embeddings_wordbase_True"]:
            for cls_type in ["avg", "max"]:
                if os.path.exists(f"images/emb/{emb_source}_xy_{model}_{cls_type}_umap_100.png"):
                    continue

                emb_folder = f"{DATA_PATH}/turn_embeddings_{emb_source}_{model}"
                print(f"{model} --- {emb_source} --- {cls_type}")
                if not os.path.exists(emb_folder):
                    continue

                label2_list = []
                df_dict = []
                for filename in os.listdir(emb_folder):
                    f = os.path.join(emb_folder, filename)

                    exp, label, label2 = None, None, None
                    if emb_source == "questionary_embeddings":
                        exp, person_id, question, _ = filename.split("_",3)
                        label = person_id_to_restriction_map[person_id]
                        label2 = question
                    elif emb_source == "transcription_embeddings":
                        exp, role, _ = filename.split("_",2)
                        if not EXPERIMENT_REST[int(exp) - 1]:
                            role = "-1"
                        label = role
                        label2 = ""
                    elif emb_source == "turn_embeddings_wordbase_False" or emb_source == "turn_embeddings_wordbase_True":
                        exp, turn_id, personx_id, persony_id, _ = filename.split("_",4)
                        if not EXPERIMENT_REST[int(exp) - 1]:
                            personx_id = "-1"
                            persony_id = "-1"
                        label = str(personx_id) + "_" + str(persony_id)
                        label2 = ""


                    emb = np.load(f)
                    if cls_type == "first":
                        emb = emb[0]
                    elif cls_type == "avg":
                        emb = np.mean(emb, axis=0)
                    elif cls_type == "max":
                        emb = np.max(emb, axis=0)

                    df_dict.append({"emb": emb, "type_name": str(label), "label2": str(label2)})
                    label2_list.append(str(label2))

                red_models = get_new_models()

                data_df = pd.DataFrame(df_dict)
                for red_model in tqdm(red_models):
                    tmp_emb_stack = np.stack(data_df["emb"], axis=0)
                    np_all_embedded = red_model["model"].fit_transform(tmp_emb_stack)
                    sns_df = pd.DataFrame({'x': np_all_embedded[:, 0], 'y': np_all_embedded[:, 1],
                                            'type_name': data_df["type_name"]})#, 'label2': label2_list})
                    sns_df = sns_df.sort_values(by=["type_name"])

                    ax = sns.scatterplot(data=sns_df, x="x", y="y", hue="type_name")#, style="label2")
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                    plt.title(red_model["name"])
                    plt.savefig(f"images/emb/{emb_source}_xy_{model}_{cls_type}_{red_model['name']}.png", dpi=300, bbox_inches='tight')
                    plt.clf()
