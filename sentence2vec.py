#! /usr/bin/python3
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
import json


def get_feature_extractor( module_path = 'https://tfhub.dev/google/universal-sentence-encoder/2'):
    '''
    Args:
        module_path: tensorflow hub link to model of our choosing
    Returns:
        jpeg_data_placeholder: tensor, we should give it result of tf.read('target_image.jpeg')
        feature_vector: our blackbox output
    '''
    module = hub.Module(module_path)

    input_tensor = tf.placeholder(tf.string)
    feature_vector = module(input_tensor)
    return input_tensor, feature_vector

with open('./Data/new_data/db.json') as f:
    memes = json.load(f)

feature_vectors = './Data/new_data/sentence_feature_vectors'
input_tensor, feature_vector = get_feature_extractor()
brojac =0
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for meme in memes.values():
        #print('\n\n\n\n' + text + '\n\n\n\n')
        try:
            print('.', end='')
            fv = sess.run(feature_vector, feed_dict={input_tensor: [ meme['ocr'] ]})
            np.savetxt(os.path.join(feature_vectors, meme['id']+'.txt'), fv)
        except:
            print(meme['id'])
            brojac += 1
print(brojac)