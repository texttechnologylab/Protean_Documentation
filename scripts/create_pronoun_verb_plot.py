import os
import pandas as pd
import matplotlib.pyplot as plt

import gzip
from ast import literal_eval
from tqdm import tqdm
import seaborn as sns
from cassis import *
from statannotations.Annotator import Annotator

DATA_PATH = "exp_data/export"

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = 20, 6
RENAME_RESTRICTION = {0: "No Restriction", 1: "Vision Restriction", 2: "Audio Restriction", 3: "Interaction Restriction"}
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


EXPERIMENT_REST = [False, True, True, True, False, True, True, False, True, True, True, True, False, False, True]



pos_pairs = [
    #[('PDS - Plur - None', 'No Restriction'), ('PDS - Plur - None', 'Vision Restriction')],
    #[('PDS - Plur - None', 'No Restriction'), ('PDS - Plur - None', 'Audio Restriction')],
    #[('PDS - Plur - None', 'No Restriction'), ('PDS - Plur - None', 'Interaction Restriction')],

    #[('PDS - Sing - None', 'No Restriction'), ('PDS - Sing - None', 'Vision Restriction')],
    #[('PDS - Sing - None', 'No Restriction'), ('PDS - Sing - None', 'Audio Restriction')],
    #[('PDS - Sing - None', 'No Restriction'), ('PDS - Sing - None', 'Interaction Restriction')],

    #[('PIS - None - None', 'No Restriction'), ('PIS - None - None', 'Vision Restriction')],
    #[('PIS - None - None', 'No Restriction'), ('PIS - None - None', 'Audio Restriction')],
    #[('PIS - None - None', 'No Restriction'), ('PIS - None - None', 'Interaction Restriction')],

    #[('PIS - Plur - None', 'No Restriction'), ('PIS - Plur - None', 'Vision Restriction')],
    #[('PIS - Plur - None', 'No Restriction'), ('PIS - Plur - None', 'Audio Restriction')],
    #[('PIS - Plur - None', 'No Restriction'), ('PIS - Plur - None', 'Interaction Restriction')],

    #[('PIS - Sing - None', 'No Restriction'), ('PIS - Sing - None', 'Vision Restriction')],
    #[('PIS - Sing - None', 'No Restriction'), ('PIS - Sing - None', 'Audio Restriction')],
    #[('PIS - Sing - None', 'No Restriction'), ('PIS - Sing - None', 'Interaction Restriction')],

    #[('PPER - Plur - 1', 'No Restriction'), ('PPER - Plur - 1', 'Vision Restriction')],
    #[('PPER - Plur - 1', 'No Restriction'), ('PPER - Plur - 1', 'Audio Restriction')],
    #[('PPER - Plur - 1', 'No Restriction'), ('PPER - Plur - 1', 'Interaction Restriction')],

    #[('PPER - Plur - 2', 'No Restriction'), ('PPER - Plur - 2', 'Vision Restriction')],
    #[('PPER - Plur - 2', 'No Restriction'), ('PPER - Plur - 2', 'Audio Restriction')],
    #[('PPER - Plur - 2', 'No Restriction'), ('PPER - Plur - 2', 'Interaction Restriction')],

    #[('PPER - Plur - 3', 'No Restriction'), ('PPER - Plur - 3', 'Vision Restriction')],
    #[('PPER - Plur - 3', 'No Restriction'), ('PPER - Plur - 3', 'Audio Restriction')],
    #[('PPER - Plur - 3', 'No Restriction'), ('PPER - Plur - 3', 'Interaction Restriction')],

    #[('PPER - Sing - 1', 'No Restriction'), ('PPER - Sing - 1', 'Vision Restriction')],
    #[('PPER - Sing - 1', 'No Restriction'), ('PPER - Sing - 1', 'Audio Restriction')],
    [('PPER - Sing - 1', 'No Restriction'), ('PPER - Sing - 1', 'Interaction Restriction')],

    #[('PPER - Sing - 2', 'No Restriction'), ('PPER - Sing - 2', 'Vision Restriction')],
    #[('PPER - Sing - 2', 'No Restriction'), ('PPER - Sing - 2', 'Audio Restriction')],
    #[('PPER - Sing - 2', 'No Restriction'), ('PPER - Sing - 2', 'Interaction Restriction')],

    #[('PPER - Sing - 3', 'No Restriction'), ('PPER - Sing - 3', 'Vision Restriction')],
    #[('PPER - Sing - 3', 'No Restriction'), ('PPER - Sing - 3', 'Audio Restriction')],
    #[('PPER - Sing - 3', 'No Restriction'), ('PPER - Sing - 3', 'Interaction Restriction')],

    #[('PWS - Sing - None', 'No Restriction'), ('PWS - Sing - None', 'Vision Restriction')],
    #[('PWS - Sing - None', 'No Restriction'), ('PWS - Sing - None', 'Audio Restriction')],
    #[('PWS - Sing - None', 'No Restriction'), ('PWS - Sing - None', 'Interaction Restriction')],
]

verb_pairs = [
    #[('gehen', 'No Restriction'), ('gehen', 'Vision Restriction')],
    #[('gehen', 'No Restriction'), ('gehen', 'Audio Restriction')],
    #[('gehen', 'No Restriction'), ('gehen', 'Interaction Restriction')],

    #[('glauben', 'No Restriction'), ('glauben', 'Vision Restriction')],
    #[('glauben', 'No Restriction'), ('glauben', 'Audio Restriction')],
    #[('glauben', 'No Restriction'), ('glauben', 'Interaction Restriction')],

    #[('greifen', 'No Restriction'), ('greifen', 'Vision Restriction')],
    [('greifen', 'No Restriction'), ('greifen', 'Audio Restriction')],
    [('greifen', 'No Restriction'), ('greifen', 'Interaction Restriction')],

    [('gucken', 'No Restriction'), ('gucken', 'Vision Restriction')],
    [('gucken', 'No Restriction'), ('gucken', 'Audio Restriction')],
    #[('gucken', 'No Restriction'), ('gucken', 'Interaction Restriction')],

    #[('helfen', 'No Restriction'), ('helfen', 'Vision Restriction')],
    [('helfen', 'No Restriction'), ('helfen', 'Audio Restriction')],
    #[('helfen', 'No Restriction'), ('helfen', 'Interaction Restriction')],

    #[('kommen', 'No Restriction'), ('kommen', 'Vision Restriction')],
    #[('kommen', 'No Restriction'), ('kommen', 'Audio Restriction')],
    #[('kommen', 'No Restriction'), ('kommen', 'Interaction Restriction')],

    #[('sehen', 'No Restriction'), ('sehen', 'Vision Restriction')],
    #[('sehen', 'No Restriction'), ('sehen', 'Audio Restriction')],
    #[('sehen', 'No Restriction'), ('sehen', 'Interaction Restriction')],

    #[('stehen', 'No Restriction'), ('stehen', 'Vision Restriction')],
    #[('stehen', 'No Restriction'), ('stehen', 'Audio Restriction')],
    #[('stehen', 'No Restriction'), ('stehen', 'Interaction Restriction')],

    [('tragen', 'No Restriction'), ('tragen', 'Vision Restriction')],
    #[('tragen', 'No Restriction'), ('tragen', 'Audio Restriction')],
    #[('tragen', 'No Restriction'), ('tragen', 'Interaction Restriction')],

    #[('warten', 'No Restriction'), ('warten', 'Vision Restriction')],
    #[('warten', 'No Restriction'), ('warten', 'Audio Restriction')],
    [('warten', 'No Restriction'), ('warten', 'Interaction Restriction')],

    #[('weiß', 'No Restriction'), ('weiß', 'Vision Restriction')],
    #[('weiß', 'No Restriction'), ('weiß', 'Audio Restriction')],
    #[('weiß', 'No Restriction'), ('weiß', 'Interaction Restriction')],
]

def load_sentiment_xmi_to_df():
    xmi_folder = "xmi_transcript/xmi"
    sentiment_list = ["sentiment_huggingface_cardiffnlp_twitter-xlm-roberta-base-sentiment",
                      "sentiment_huggingface_LiYuan_amazon-review-sentiment-analysis",
                      "sentiment_huggingface_mdraw_german-news-sentiment-bert",
                      "sentiment_huggingface_nlptown_bert-base-multilingual-uncased-sentiment",
                      "sentiment_huggingface_oliverguhr_german-sentiment-bert",
                      "sentiment_vader",
                      "sentiment_textblob"
                      ]

    with open('TypeSystem.xml', 'rb') as f:
        typesystem = load_typesystem(f)

    df_list = []
    for foldername in tqdm(sentiment_list):
        for filename in os.listdir(os.path.join(xmi_folder, foldername)):
            if filename == "TypeSystem.xml.gz":
                continue
            exp = int(filename.split("_")[0]) - 1
            role = int(filename.split("_")[1][-1])
            user_id = int(filename[-12])
            user_name = f"{exp}_{role}_{user_id}"

            restriction = 0 if not EXPERIMENT_REST[exp] else role
            # restriction = RENAME_RESTRICTION[restriction]
            gz_path = os.path.join(xmi_folder, foldername, filename)
            with gzip.open(gz_path, 'rb') as f_in:
                # print(gz_path)
                cas = load_cas_from_xmi(f_in, typesystem=typesystem)
                # print(cas)

                for sentence in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"):
                    token_list = []
                    lemma_list = []
                    pos_list = []
                    pos_fine_list = []
                    cas_list = []
                    number_list = []
                    person_list = []

                    sentiment = 0
                    for token in cas.select_covered("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", sentence):
                        token_list.append(token.get_covered_text())
                        lemma_list.append(token.lemma.value)
                        pos_list.append(token.pos.coarseValue)
                        pos_fine_list.append(token.pos.PosValue)


                    for sentiment in cas.select_covered("org.hucompute.textimager.uima.type.CategorizedSentiment", sentence):
                        sentiment = sentiment.sentiment

                    for morph in cas.select_covered \
                            ("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures", sentence):
                        cas_list.append(morph.case)
                        number_list.append(morph.number)
                        person_list.append(morph.person)

                    df_list.append({"experiment": exp, "person": role, "restriction": restriction,
                                    "sentence_start": sentence.begin, "sentence_end": sentence.end,
                                    "sentence_list": token_list, "lemma_list": lemma_list,
                                    "pos_list": pos_list, "pos_fine_list": pos_fine_list,
                                    "case_list": cas_list, "number_list": number_list, "person_list": person_list,
                                    "sentiment": sentiment, "user_id": user_name})

    df = pd.DataFrame(df_list)
    df.to_csv(f"{DATA_PATH}/transcription_sentiment.csv")
    df_without_sentence = df.drop \
        (columns=["user_id", "sentence_list", "pos_list", "case_list", "number_list", "person_list", "lemma_list", "pos_fine_list"])
    # df = df.drop(columns=["sentiment_tagger"])
    df_without_sentence_grouped = df_without_sentence.groupby(
        ['experiment', 'person', "restriction", "sentence_start", "sentence_end"])

    df_without_sentence_grouped.mean().to_csv(f"{DATA_PATH}/transcription_sentiment_grouped.csv")


def create_sentiment_pron_graph():
    token_count_list = [0, 0, 0, 0]
    user_token_count = {}
    df = pd.read_csv(f"{DATA_PATH}/transcription_sentiment.csv", converters={"sentence_list": literal_eval, "lemma_list": literal_eval,
                                                  "pos_list": literal_eval, "pos_fine_list": literal_eval,
                                                  "case_list": literal_eval, "number_list": literal_eval, "person_list": literal_eval})
    df_without_sentence_grouped = pd.read_csv(f"{DATA_PATH}/transcription_sentiment_grouped.csv")

    result_list = []
    filter = 0.1
    with_pos = True
    for index, row in df_without_sentence_grouped.iterrows():

        sent = df.loc[(df["sentence_start"] == row["sentence_start"]) &
                           (df["sentence_end"] == row["sentence_end"]) &
                           (df["experiment"] == row["experiment"]) &
                           (df["person"] == row["person"])]

        sent = sent.iloc[0]
        token_list = sent["sentence_list"]
        lemma_list = sent["lemma_list"]
        pos_list = sent["pos_list"]
        pos_fine_list = sent["pos_fine_list"]
        case_list = sent["case_list"]
        number_list = sent["number_list"]
        person_list = sent["person_list"]
        token_count_list[int(row["restriction"])] += len(token_list)
        user_id = sent["user_id"]
        if user_id not in user_token_count.keys():
            user_token_count[user_id] = len(token_list)
        else:
            user_token_count[user_id] += len(token_list)

        for token, lemma, pos, pos_fine, case, number, person in zip(token_list, lemma_list, pos_list, pos_fine_list, case_list, number_list, person_list):
            if with_pos:
                if pos_fine in ["ADJD", "ADV","PRF", "PRELS", "ART", "CARD", "NE", "NN", "PIAT", "PPOSAT", "FM", "VVFIN"]:
                    continue
                if pos_fine == "PPER" and (number is None or person is None):
                    continue
                if pos_fine == "PDS" and number is None:
                    continue
                if pos_fine == "PWS" and number is None:
                    continue
            if pos == "PRON": # and token.lower() in PRONOUN_LIST:
                #print(token, lemma, pos, pos_fine, case, number, person)
                if row["sentiment"] > filter:
                    if with_pos:
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{pos_fine} - {number} - {person}", "count": 1,
                             "sentiment": "pos"})
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{pos_fine} - {number} - {person}", "count": 0,
                             "sentiment": "neg"})
                    else:
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{number} - {person}", "count": 1,
                             "sentiment": "pos"})
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{number} - {person}", "count": 0,
                             "sentiment": "neg"})

                elif row["sentiment"] < -filter:
                    if with_pos:
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{pos_fine} - {number} - {person}", "count": 1,
                             "sentiment": "neg"})
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{pos_fine} - {number} - {person}", "count": 0,
                             "sentiment": "pos"})
                    else:
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{number} - {person}", "count": 1,
                             "sentiment": "neg"})
                        result_list.append(
                            {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{number} - {person}", "count": 0,
                             "sentiment": "pos"})

    df = pd.DataFrame(result_list)

    df = df.groupby(["user_id", "restriction", "pronoun", "sentiment"]).sum().reset_index()
    max_list_idx = list(df["user_id"].tolist())
    max_list = [user_token_count[x] for x in max_list_idx]
    df["max"] = max_list
    df["percent"] = df["count"] / df["max"]
    df = df.sort_values(by=["pronoun", "restriction"])

    df["restriction"] = df["restriction"].apply(
        lambda x: RENAME_RESTRICTION[x] if x in RENAME_RESTRICTION.keys() else x)

    df['sentiment'] = pd.Categorical(df['sentiment'])  # make hue column categorical, forcing a fixed order


    sns.set_theme(style='whitegrid')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), sharey=True, gridspec_kw={'wspace': 0})

    # draw adult subplot at the right
    sns.barplot(data=df[df['sentiment'] == 'pos'], x='percent', y='pronoun', hue='restriction', orient='horizontal', dodge=True, ax=ax2, errwidth=2)
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', labelright=True, right=True)
    ax2.set_title('  ' + 'pos', loc='left')
    ax2.legend_.remove()  # remove the legend; the legend will be in ax1

    # draw juvenile subplot at the left
    sns.barplot(data=df[df['sentiment'] == 'neg'], x='percent', y='pronoun', hue='restriction', orient='horizontal', dodge=True, ax=ax1, errwidth=2)

    # optionally use the same scale left and right
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ax1.set_xlim(xmax=xmax)
    ax2.set_xlim(xmax=xmax)

    ax1.invert_xaxis()  # reverse the direction
    ax1.tick_params(labelleft=False, left=False)
    ax1.set_ylabel('')
    ax1.set_title('neg' + '  ', loc='right')

    annotator = Annotator(ax2, pos_pairs, data=df, x='percent', y='pronoun', hue='restriction', orient='h', dodge=True, errwidth=2, plot='barplot')
    annotator.configure(test="Kruskal", text_format="simple", show_test_name=False).apply_and_annotate()

    #annotator = Annotator(ax1, pos_pairs, data=df[df['sentiment'] == 'neg'], x='percent', y='pronoun', hue='restriction', orient='h', dodge=True, errwidth=2, plot='barplot')
    #annotator.configure(test="Kruskal", text_format="simple", show_test_name=False).apply_and_annotate()

    plt.tight_layout()
    plt.savefig(f"results/sentiment_pos_{with_pos}_number_person_{filter}.png", dpi=300)
    plt.show()



def create_sentiment_verb_graph():

    token_count_list = [0, 0, 0, 0]
    user_token_count = {}
    df = pd.read_csv("sentiment_error_icmi.csv", converters={"sentence_list": literal_eval, "lemma_list": literal_eval,
                                                  "pos_list": literal_eval, "pos_fine_list": literal_eval,
                                                  "case_list": literal_eval, "number_list": literal_eval, "person_list": literal_eval})
    df_without_sentence_grouped = pd.read_csv("sentiment_mean_group_error_icmi.csv")

    result_list = []
    filter = 0.1

    for index, row in df_without_sentence_grouped.iterrows():
        if row["experiment"] == 0 or row["experiment"] == 5:
            continue
        if True:

            sent = df.loc[(df["sentence_start"] == row["sentence_start"]) &
                               (df["sentence_end"] == row["sentence_end"]) &
                               (df["experiment"] == row["experiment"]) &
                               (df["person"] == row["person"])]

            sent = sent.iloc[0]
            token_list = sent["sentence_list"]
            lemma_list = sent["lemma_list"]
            pos_list = sent["pos_list"]
            pos_fine_list = sent["pos_fine_list"]
            case_list = sent["case_list"]
            number_list = sent["number_list"]
            person_list = sent["person_list"]
            token_count_list[int(row["restriction"])] += len(token_list)
            user_id = sent["user_id"]
            if user_id not in user_token_count.keys():
                user_token_count[user_id] = len(token_list)
            else:
                user_token_count[user_id] += len(token_list)

            for token, lemma, pos, pos_fine, case, number, person in zip(token_list, lemma_list, pos_list, pos_fine_list, case_list, number_list, person_list):
                if pos == "VERB": # and token.lower() in PRONOUN_LIST:
                    print(token, lemma, pos, pos_fine, case, number, person)
                    if row["sentiment"] > filter:
                        result_list.append(
                                {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{lemma.lower()}", "count": 1,
                                 "sentiment": "pos"})
                        result_list.append(
                                {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{lemma.lower()}", "count": 0,
                                 "sentiment": "neg"})

                    elif row["sentiment"] < -filter:
                        result_list.append(
                                {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{lemma.lower()}", "count": 1,
                                 "sentiment": "neg"})
                        result_list.append(
                                {"user_id": user_id, "restriction": row["restriction"], "pronoun": f"{lemma.lower()}", "count": 0,
                                 "sentiment": "pos"})

                    result_list.append(
                        {"user_id": user_id, "restriction": 0, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "neg"})
                    result_list.append(
                        {"user_id": user_id, "restriction": 1, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "neg"})
                    result_list.append(
                        {"user_id": user_id, "restriction": 2, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "neg"})
                    result_list.append(
                        {"user_id": user_id, "restriction": 3, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "neg"})

                    result_list.append(
                        {"user_id": user_id, "restriction": 0, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "pos"})
                    result_list.append(
                        {"user_id": user_id, "restriction": 1, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "pos"})
                    result_list.append(
                        {"user_id": user_id, "restriction": 2, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "pos"})
                    result_list.append(
                        {"user_id": user_id, "restriction": 3, "pronoun": f"{lemma.lower()}", "count": 0,
                         "sentiment": "pos"})

    print("..................")
    df = pd.DataFrame(result_list)
    df = df.groupby(["user_id", "restriction", "pronoun", "sentiment"]).sum().reset_index()

    FILTER_VERBS = ["glauben", "helfen", "gehen", "sehen", "stehen", "warten", "weiß", "greifen", "kommen", "gucken", "tragen"]
    #FILTER_VERBS = ["glauben", "gehen", "sehen", "stehen", "warten", "weiß", "greifen", "kommen", "gucken",
    #                "tragen"]

    df = df[df["pronoun"].isin(FILTER_VERBS)]

    #df = df.sort_values(by=["count"], ascending=False)


    max_list_idx = list(df["user_id"].tolist())
    max_list = [user_token_count[x] for x in max_list_idx]
    df["max"] = max_list
    df["percent"] = df["count"] / df["max"]
    df = df.sort_values(by=["pronoun", "restriction"])

    df["restriction"] = df["restriction"].apply(
        lambda x: RENAME_RESTRICTION[x] if x in RENAME_RESTRICTION.keys() else x)

    df['sentiment'] = pd.Categorical(df['sentiment'])  # make hue column categorical, forcing a fixed order

    df = df.rename(columns={"pronoun": "verb"})


    sns.set_theme(style='whitegrid')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), sharey=True, gridspec_kw={'wspace': 0})


    # draw adult subplot at the right
    sns.barplot(data=df[df['sentiment'] == 'pos'], x='percent', y='verb', hue='restriction', orient='horizontal', dodge=True, ax=ax2, errwidth=2)
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', labelright=True, right=True)
    ax2.set_title('  ' + 'pos', loc='left')
    ax2.legend_.remove()  # remove the legend; the legend will be in ax1

    # draw juvenile subplot at the left
    sns.barplot(data=df[df['sentiment'] == 'neg'], x='percent', y='verb', hue='restriction', orient='horizontal', dodge=True, ax=ax1, errwidth=2)

    # optionally use the same scale left and right
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ax1.set_xlim(xmax=xmax)
    ax2.set_xlim(xmax=xmax)


    ax1.invert_xaxis()  # reverse the direction
    ax1.tick_params(labelleft=False, left=False)
    ax1.set_ylabel('')
    ax1.set_title('neg' + '  ', loc='right')

    import matplotlib.ticker as ticker
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.0005))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.0005))

    annotator = Annotator(ax2, verb_pairs, data=df, x='percent', y='verb', hue='restriction', orient='h', dodge=True, errwidth=2, plot='barplot')
    annotator.configure(test="Kruskal", text_format="simple", show_test_name=False).apply_and_annotate()


    plt.tight_layout()
    plt.savefig(f"results/sentiment_verbs.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # Sentiments for transcriptions were created with *reference follows for anonymization*
    load_sentiment_xmi_to_df()

    create_sentiment_pron_graph()
    create_sentiment_verb_graph()