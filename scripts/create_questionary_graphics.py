import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from statannotations.Annotator import Annotator

from help_function import read_json, flat_list

DATA_PATH = "exp_data/export"

EXPERIMENT_REST = [False, True, True, True, False, True, True, False, True, True, True, True, False, False, True]

FREE_TEXT_LABELS = {"BS04_01": "Controls", "BS05_01": "Interfaces", "BI03_01": "User Interaction", "BI04_01": "Avatar",
                    "WB02_01": "Wellbeing", "SZ04_01": "Tutorial", "SZ06_01": "VR Experience",
                    "SZ03_01": "Szenario"}  # , "SD18_01": "Additional Comments"}

RESTRICTION_LABELS = {-1: "No Restriction", 1: "Vision Restriction", 2: "Audio Restriction",
                      3: "Interaction Restriction"}

CATEGORY_DICT = {"BS01": "UMUX", "BS02": "Controls", "BS03": "Interfaces",
                 "BI01": "User Interaction", "BI02": "Avatar", "WB01": "Wellbeing",
                 "SZ02": "Tutorial", "SZ05": "VR Experiance", "SZ01": "Scenario"}

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



def create_df_from_json(json_merge, invert=False):
    df_data = []
    for experiment in json_merge.keys():
        for person in json_merge[experiment].keys():
            person_data = json_merge[experiment][person]
            person_dict = {"experiment": int(experiment), "person_id": person, "person_role": person_data["DI01"]}
            person_dict["restriction"] = -1 if not EXPERIMENT_REST[int(experiment) - 1] else person_data["DI01"]
            person_dict["feeled_restriction"] = person_data["DI02"] - 2 if person_data["DI02"] == 1 else person_data[
                                                                                                             "DI02"] - 1

            person_dict["started"] = person_data["STARTED"]
            # print(person_data["DI01"])
            for score in EVAL_SCORES_FLAT:
                if invert and score in INV_LIST and person_data[score] is not None and person_data[score] >= 0:
                    person_dict[score] = 6 - person_data[score]  # skale is from 1-5 ...
                else:
                    person_dict[score] = person_data[score]

            for free_text in FREE_TEXT_LABELS.keys():
                # print("!!!!!")
                # print(free_text)
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


json_merge = read_json(f"{DATA_PATH}/all_simulation_evaluation_infos_sentiment.json")

# df = create_df_from_json(json_merge)
df_inv = create_df_from_json(json_merge, invert=True)


# for every colum in eval_scores_flat, set -1 to None
for score in EVAL_SCORES_FLAT:
    df_inv[score] = df_inv[score].apply(lambda x: None if x == -1 else x)


def generate_df_for_merged_box_plot_avg(df, column_names):
    plotdf = df[column_names + ['restriction']]
    #display(HTML(plotdf.to_html()))

    for eval_score_group in EVAL_SCORES:
        groupname = eval_score_group[0].split("_")[0]
        tmp_df = df[eval_score_group]
        plotdf[groupname + "_avg"] = tmp_df.mean(axis=1, skipna=True)



    avg_colums = [col for col in plotdf if col.endswith('_avg')]

    plotdf = plotdf[avg_colums + ['restriction']]
        #for score in eval_score_group:

    plotdf = plotdf.melt(id_vars=['restriction'], value_vars=avg_colums)
    plotdf = plotdf.drop(plotdf[plotdf["value"] < 0].index)

    plotdf[['CAT', 'QUEST']] = plotdf["variable"].str.split("_", expand=True)

    plotdf["CAT"] = plotdf["CAT"].apply(lambda x: CATEGORY_DICT[x] if x in CATEGORY_DICT.keys() else x)
    plotdf = plotdf.sort_values(by=["restriction", "CAT"])
    plotdf["restriction"] = plotdf["restriction"].apply(
        lambda x: RESTRICTION_LABELS[x] if x in RESTRICTION_LABELS.keys() else x)
    return plotdf


plot_df = generate_df_for_merged_box_plot_avg(df_inv, EVAL_SCORES_FLAT)

pairs = [
    [('Avatar', 'No Restriction'), ('Avatar', 'Vision Restriction')],
    [('Avatar', 'No Restriction'), ('Avatar', 'Audio Restriction')],
    [('Avatar', 'No Restriction'), ('Avatar', 'Interaction Restriction')],

    [('Controls', 'No Restriction'), ('Controls', 'Vision Restriction')],
    [('Controls', 'No Restriction'), ('Controls', 'Audio Restriction')],
    [('Controls', 'No Restriction'), ('Controls', 'Interaction Restriction')],

    [('Interfaces', 'No Restriction'), ('Interfaces', 'Vision Restriction')],
    [('Interfaces', 'No Restriction'), ('Interfaces', 'Audio Restriction')],
    [('Interfaces', 'No Restriction'), ('Interfaces', 'Interaction Restriction')],

    [('Scenario', 'No Restriction'), ('Scenario', 'Vision Restriction')],
    [('Scenario', 'No Restriction'), ('Scenario', 'Audio Restriction')],
    [('Scenario', 'No Restriction'), ('Scenario', 'Interaction Restriction')],

    [('Tutorial', 'No Restriction'), ('Tutorial', 'Vision Restriction')],
    [('Tutorial', 'No Restriction'), ('Tutorial', 'Audio Restriction')],
    [('Tutorial', 'No Restriction'), ('Tutorial', 'Interaction Restriction')],

    [('User Interaction', 'No Restriction'), ('User Interaction', 'Vision Restriction')],
    [('User Interaction', 'No Restriction'), ('User Interaction', 'Audio Restriction')],
    [('User Interaction', 'No Restriction'), ('User Interaction', 'Interaction Restriction')],

    [('VR Experiance', 'No Restriction'), ('VR Experiance', 'Vision Restriction')],
    [('VR Experiance', 'No Restriction'), ('VR Experiance', 'Audio Restriction')],
    [('VR Experiance', 'No Restriction'), ('VR Experiance', 'Interaction Restriction')],

    [('Wellbeing', 'No Restriction'), ('Wellbeing', 'Vision Restriction')],
    [('Wellbeing', 'No Restriction'), ('Wellbeing', 'Audio Restriction')],
    [('Wellbeing', 'No Restriction'), ('Wellbeing', 'Interaction Restriction')],
]

# "CAT", "value", "restriction", "Evaluation Scores"
hue_plot_params = {
    'data': plot_df,
    'x': "CAT",
    'y': "value",
    # "order": subcat_order,
    "hue": "restriction",
    # "hue_order": states_order,
    # "palette": state_palette
}

# Plot with seaborn
plt.figure(figsize=(20, 6))
ax = sns.boxplot(**hue_plot_params)

# Add annotations
annotator = Annotator(ax, pairs, **hue_plot_params)
# ValueError: t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal, Brunner-Munzel.
annotator.configure(test="Kruskal", text_format="simple", show_test_name=False).apply_and_annotate()


plt.savefig(f"images/evaluation_boxplot_avg.png", dpi=300, transparent=True)
plt.show()


def generate_df_for_sentiment_avg_box_plot(df):
    freetext_colums = [col for col in df if col.startswith('sentavg:')]
    plotdf = df[freetext_colums + ['restriction']]
    plotdf = plotdf.melt(id_vars=['restriction'], value_vars=freetext_colums)
    plotdf[['sent', 'CAT']] = plotdf["variable"].str.split(":", expand=True)
    plotdf["CAT"] = plotdf["CAT"].apply(lambda x: FREE_TEXT_LABELS[x] if x in FREE_TEXT_LABELS.keys() else x)
    plotdf = plotdf.sort_values(by=["restriction", "CAT"])
    plotdf = plotdf.dropna()
    plotdf["restriction"] = plotdf["restriction"].apply(
        lambda x: RESTRICTION_LABELS[x] if x in RESTRICTION_LABELS.keys() else x)
    return plotdf

sent_plot_df = generate_df_for_sentiment_avg_box_plot(df_inv)
sent_plot_df = sent_plot_df.replace("Szenario", "Scenario")

hue_plot_params = {
    'data': sent_plot_df,
    'x': "CAT",
    'y': "value",
    "hue": "restriction",
}


plt.figure(figsize=(20, 6))
ax = sns.boxplot(**hue_plot_params)
sns.move_legend(ax, "lower right")

# Add annotations
annotator = Annotator(ax, pairs, **hue_plot_params)
#ValueError: t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal, Brunner-Munzel.
annotator.configure(test="Kruskal", text_format="simple", show_test_name=False).apply_and_annotate()


plt.savefig(f"images/evaluation_sentiment_boxplot_avg.png", dpi=300, transparent=True)
plt.show()