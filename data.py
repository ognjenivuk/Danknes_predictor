import os
import json
import math

def read_vectors_from_files(file_dir):
    encodings = {}
    for vector_file in os.listdir(file_dir):
        with open(os.path.join(file_dir, vector_file), 'r') as f:
            encodings[os.path.splitext(vector_file)[0]] = [float(x) for x in f.read().split(' ')]

    return encodings

def load_scores(json_file):
        
    with open(json_file) as f:
        json_file_loaded = json.load(f)

    scores = {v['id'] : math.log(v['ups']) for v in json_file_loaded['_default'].values()}

    return scores