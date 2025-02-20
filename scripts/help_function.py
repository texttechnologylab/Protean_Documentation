import os
import json
import gzip

import numpy as np

def argmin(lst):
    return lst.index(min(lst))

def flat_list(l):
    return [item for sublist in l for item in sublist]

def save_json(json_data, data_dir, gzip_save=False):
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    if gzip_save:
        with gzip.open(data_dir, "wt", encoding="UTF-8") as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)
    else:
        with open(data_dir, "w", encoding="UTF-8") as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)


def read_json(data_dir, gzip_load=False):
    if gzip_load:
        with gzip.open(data_dir, "rt", encoding="UTF-8") as json_file:
            return json.load(json_file)
    else:
        with open(data_dir, "r", encoding="UTF-8") as json_file:
            return json.load(json_file)


def compute_euclidean_distance(array1, array2):
    return np.linalg.norm(array1 - array2)