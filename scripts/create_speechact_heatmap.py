import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import colormaps
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from tqdm import tqdm
import seaborn as sns

DATA_PATH = "exp_data/export"


MODELS = [
    "gemma-2-9b-it-Q5_K_M.gguf",
    "gemma-2-27b-it-Q5_K_M.gguf",
    "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
    "Mistral-Nemo-Instruct-2407-Q5_K_M.gguf",
    "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
    "c4ai-command-r-08-2024-Q5_K_M.gguf",
    "Llama-3.1-Storm-8B.Q5_K_M.gguf",
]


DIALOG_CLASSES_EN = {
    "1": "Task (or Activity)",
    "2": "Auto-and Allo-Feedback",
    "3": "Turn Management",
    "4": "Time Management",
    "5": "Discourse Structuring",
    "6": "Own and Partner Communication Management",
    "7": "Social Obligations Management",
    "8": "Broken",
}
DIALOG_CLASSES_EN_INV = {
    v: k
    for k, v in DIALOG_CLASSES_EN.items()
}

DIALOG_CLASSES_DE = {
    "1": "Aufgabe (oder Aktivität)",
    "2": "Auto- und Allo-Feedback",
    "3": "Gesprächssteuerung",
    "4": "Zeitmanagement",
    "5": "Diskursstrukturierung",
    "6": "Eigene und Partner-Kommunikationsverwaltung",
    "7": "Verwaltung sozialer Verpflichtungen",
    "8": "Defekt",
}
DIALOG_CLASSES_DE_INV = {
    v: k
    for k, v in DIALOG_CLASSES_EN.items()
}

DIALOG_CLASSES_DE_ALT = {
    "1": "Frage (oder Aktivität)",
    "2": "Aussage/Information",
    "3": "Aufforderung",
    "4": "Bestätigung/Verneinung",
    "5": "Sozialer Akt",
    "6": "Defekt",
}
DIALOG_CLASSES_DE_ALT_INV = {
    v: k
    for k, v in DIALOG_CLASSES_EN.items()
}


DIALOG_CLASSES_EN_ALT = {
    "1": "Question (or Activity)",
    "2": "Statement/Information",
    "3": "Request",
    "4": "Confirmation/Denial",
    "5": "Social Act",
    "6": "Defect",
}
DIALOG_CLASSES_EN_ALT_INV = {
    v: k
    for k, v in DIALOG_CLASSES_EN.items()
}

RESTRICTION_LABELS = {-1: "NoRe", 1: "Vision", 2: "Audio", 3: "Interaction"}
RENAME_EXPERIMENT = {5: 1, 8: 2, 13: 3, 14: 4, 2: 5, 3: 6, 4: 7, 7: 8, 9: 9, 10: 10, 11: 11, 12: 12, 15: 13}


def parse_label(label: str):
    pattern = r"(?:Typ|Type):\s*(.*?)[;.\s]*\s*(?:Vertrauen|Confidence):\s*([\d.,]+)[;.\s]*"
    matches = re.findall(pattern, label)
    if len(matches) == 0:
        return "None", None
    label, score = matches[0]
    score = score.replace(",", ".")
    if not score[-1].isdigit():
        score = score[:-1]
    try:
        score = float(score)
    except:
        return "None", None
    return label, float(score)


collected_dfs = {}
for model in tqdm(MODELS):
    for mode in ["turns_classifications", "wordbased_turns_classifications", "turns_alt_classifications", "wordbased_turns_alt_classifications"]:
        for promt_type in ["enpromt", "depromt"]: #  ["depromt", "enpromt"]:
            file_name = f"{DATA_PATH}/turn_prompt/transcriptions_{mode}_{promt_type}_{model}.csv"
            if not os.path.exists(file_name):
                print(f"File {file_name} not found")
                continue
            df = pd.read_csv(file_name)

            df_tmp = df.apply(lambda x: parse_label(x["sentiment"]), axis=1)
            df_parsed = pd.DataFrame(df_tmp.tolist(), index=df_tmp.index, columns=['annotation_label', 'annotation_score'])
            df_combined = df.join(df_parsed)
            collected_dfs[(model, mode, promt_type)] = df_combined



for key, df in collected_dfs.items():
    df.loc[df["exp_id"].isin([5, 8, 13, 14]), "player_x"] = -1
    df.loc[df["exp_id"].isin([5, 8, 13, 14]), "player_y"] = -1


    # create df with type count for each experiment
    df_sorted = df.groupby(["exp_id", "annotation_label"]).size().unstack(fill_value=0)

    # merge/sum "Task" and "Aufgabe" columns to "Aufgabe"
    if "Aufgabe (oder Aktivität)" in df_sorted.columns and "Aufgabe" in df_sorted.columns:
        df_sorted["Aufgabe"] = df_sorted["Aufgabe"] + df_sorted["Aufgabe (oder Aktivität)"]
        df_sorted = df_sorted.drop(columns=["Aufgabe (oder Aktivität)"])
        df_sorted = df_sorted.rename(columns={"Aufgabe": "Aufgabe (oder Aktivität)"})

    if "Task (or Activity)" in df_sorted.columns and "Task" in df_sorted.columns:
        df_sorted["Task"] = df_sorted["Task"] + df_sorted["Task (or Activity)"]
        df_sorted = df_sorted.drop(columns=["Task (or Activity)"])
        df_sorted = df_sorted.rename(columns={"Task": "Task (or Activity)"})



    # existing_columns = [col for col in DIALOG_CLASSES_EN_INV.keys() if col in df_sorted.columns] + [col for col in DIALOG_CLASSES_DE_INV.keys() if col in df_sorted.columns] #  + [col for col in ["Task", "Aufgabe"] if col in df_sorted.columns]

    # existing_columns = set(existing_columns)
    # existing_columns = sorted(existing_columns)
    # df_sorted = df_sorted[existing_columns]
    # display(df_sorted)

    # dfviz = df_sorted.T
    # dfviz.columns = [RENAME_EXPERIMENT.get(col, col) for col in dfviz.columns]
    # dfviz = dfviz.sort_index(axis=1)

    # dfviz rename colums after RESTRICTION_LABELS
    df_sorted = df.sort_values(by=["player_y", "annotation_label"])
    df_sorted = df_sorted.groupby(["player_y", "annotation_label"]).size().unstack(fill_value=0)
    # drop columns with "None" and "Time Managment"
    if "None" in df_sorted.columns:
        df_sorted = df_sorted.drop(columns=["None"])
    if "Time Management" in df_sorted.columns:
        df_sorted = df_sorted.drop(columns=["Time Management"])

    dfviz = df_sorted.T
    dfviz.columns = [RESTRICTION_LABELS.get(col, col) for col in dfviz.columns]
    dfviz["NoRe"] = dfviz["NoRe"] / (4 * 3)
    dfviz["Vision"] = dfviz["Vision"] / 9
    dfviz["Audio"] = dfviz["Audio"] / 9
    dfviz["Interaction"] = dfviz["Interaction"] / 9

    # confert float NoRe to int
    dfviz["NoRe"] = dfviz["NoRe"].astype(int)
    dfviz["Vision"] = dfviz["Vision"].astype(int)
    dfviz["Audio"] = dfviz["Audio"].astype(int)
    dfviz["Interaction"] = dfviz["Interaction"].astype(int)

    row_sums = dfviz.sum(axis=1)
    col_sums = dfviz.sum()
    col_sums = col_sums.to_frame().T
    col_sums['_name'] = 'Sum'
    col_sums.set_index('_name', inplace=True)
    dfviz2 = pd.concat([dfviz, col_sums])
    dfviz2["Sum"] = row_sums

    # display(dfviz2)

    fig, ax = plt.subplots(1, figsize=(20, 5))

    mask = np.zeros_like(dfviz2, dtype=bool)
    mask[-1, :] = True  # Mask the last row
    mask[:, -1] = True  # Mask the last column

    sns.heatmap(
        dfviz2,
        cmap='Reds',
        annot=True,
        fmt=".0f",
        square=True,
        ax=ax,
        mask=mask,
    )

    # Normalize values for the gradient
    # norm = Normalize(vmin=np.nanmin(dfviz2.iloc[:-1, :-1].values), vmax=np.nanmax(dfviz2.iloc[:-1, :-1].values))
    # norm = Normalize(vmin=0, vmax=170)
    norm = Normalize(vmin=0, vmax=1700)
    cmap = colormaps.get_cmap('Blues')

    # Overlay the sum values
    for i in range(dfviz2.shape[0] - 1):
        color = cmap(norm(dfviz2.iloc[i, -1]))
        ax.add_patch(Rectangle((dfviz2.shape[1] - 1, i), 1, 1, fill=True, color=color))
        # ax.add_patch(Rectangle((dfviz2.shape[1] - 1, i), 1, 1, fill=True, color='lightgrey'))
        color = 'white' if dfviz2.iloc[i, -1] > 500 else 'black'
        plt.text(dfviz2.shape[1] - 0.5, i + 0.5, f"{dfviz2.iloc[i, -1]:.0f}", horizontalalignment='center',
                 verticalalignment='center', color=color)
    for j in range(dfviz2.shape[1] - 1):
        color = cmap(norm(dfviz2.iloc[-1, j]))
        ax.add_patch(Rectangle((j, dfviz2.shape[0] - 1), 1, 1, fill=True, color=color))
        # ax.add_patch(Rectangle((j, dfviz2.shape[0] - 1), 1, 1, fill=True, color='lightgrey'))
        color = 'white' if dfviz2.iloc[-1, j] > 500 else 'black'
        plt.text(j + 0.5, dfviz2.shape[0] - 0.5, f"{dfviz2.iloc[-1, j]:.0f}", horizontalalignment='center',
                 verticalalignment='center', color=color)
    # ax.add_patch(Rectangle((dfviz2.shape[1] - 1, dfviz2.shape[0] - 1), 1, 1, fill=True, color='lightgrey'))

    # Add overlay gradient colorbar
    fig = plt.gcf()
    # cbar_ax = fig.add_axes([0.9, 0.17, 0.01, 0.80]) # restriction group ...
    cbar_ax = fig.add_axes([0.9, 0.07, 0.01, 0.9])
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.outline.set_visible(False)
    # cb.set_label('Sum Values')

    plt.tight_layout()
    # plt.savefig(f"images/turn_graphics/exp_heatmap_lvlgroup_{key[0]}_{key[1]}_{key[2]}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"images/turn_graphics/exp_heatmap_restgroup_{key[0]}_{key[1]}_{key[2]}.png", dpi=300,
                bbox_inches='tight')
    # plt.show()
    # plt.savefig("exp_heatmap_v3_turn2.png", dpi=300)
    # plt.savefig(datapath_classifications / f"human/{llm_name}/{window_name}/exp_heatmap_sums_v1.png", dpi=300)

