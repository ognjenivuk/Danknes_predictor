#! /usr/bin/python3
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
import json

def add_image_deocding(module):
    input_height, input_width = hub.get_expected_image_size(module)
    input_depth = hub.get_num_image_channels(module)
    input_file = tf.placeholder(tf.string, name = 'InputFile')
    file_reader = tf.read_file(input_file)
    decoded_image = tf.image.decode_png(file_reader, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                            resize_shape_as_int)
    return input_file, resized_image

def get_feature_extractor( module_path = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3'):
    module = hub.Module(module_path)
    input_file, resized_image = add_image_deocding(module)
    feature_vector = module(resized_image)
    return input_file, feature_vector

json_file = './Data/new_data/db.json'
with open(json_file) as f:
    memes = json.load(f)
brojac = 0
feature_vectors = './Data/new_data/image_feature_vectors'
input_tensor, feature_vector = get_feature_extractor('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i,image_data in enumerate(memes.values()):
        print(i, end='\r')
        try:
            feature_vector_calculated = \
            sess.run(feature_vector, feed_dict={input_tensor:os.path.join('./Data/new_data/merged', image_data['filename'].replace('.jpg', '.png'))})
            np.savetxt(os.path.join(feature_vectors, image_data['id']+'.txt'), feature_vector_calculated)
        except Exception as e: 
            print(image_data['id'])
            print(e)
            brojac += 1
print(brojac)