import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


from help_function import read_json
from wordcloud import WordCloud, STOPWORDS
from spacy.lang.de.stop_words import STOP_WORDS as STOP_WORDS_DE


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


EXPERIMENT_REST = [False, True, True, True, False, True, True, False, True, True, True, True, False, False, True]

DATA_PATH = "exp_data/export"


def generate_wordcloud_all():
    directory = f"{DATA_PATH}/audio_export_json/"
    text = ""
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            data = read_json(f)
            text += data["text"] + "\n"

    wordcloud = WordCloud(stopwords=STOP_WORDS_DE, background_color="white", width=2400, height=1200).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.savefig("results/wordcloud_all_300dpi.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    generate_wordcloud_all()

