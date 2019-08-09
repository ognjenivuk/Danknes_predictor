#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import data
import math
from model import Model

def _get_named_model(name, basedir='./trained_models/'):
    potential = [f[:-len('.index')] for f in os.listdir(basedir) if f.endswith('.cpkt.index') and f.startswith(name)]
    potential.sort()
    return os.path.join(basedir, potential[-1])


def main():
    meme_id = input('meme id : ')
    sentence_vectors = data.read_vectors_from_files('./Data/sentence_feature_vectors')
    image_vectors = data.read_vectors_from_files('./Data/image_feature_vectors')
    scores = data.load_scores('./Data/db.json')
    model = Model()
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(model.sess, _get_named_model('model_from_hub_modules'))
    print("Model restored.")
    output = model.sess.run([model.output], feed_dict={
        model.image_encoded: [ image_vectors[meme_id] ],
        model.sentence_encoded: [ sentence_vectors[meme_id] ]
    })

    print(' predicted:', math.exp(output[0][0,0]))
    print('    actual:', math.exp(scores[meme_id]))

if __name__ == "__main__":
    main()