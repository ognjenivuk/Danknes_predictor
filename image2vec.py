#! /usr/bin/python3
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os

def add_image_deocding(module):
  """Adds operations that perform JPEG decoding and resizing to the graph..
  Args:
    module_spec: The hub.ModuleSpec for the image module being used.
  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
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
    '''
    Args:
        module_path: tensorflow hub link to model of our choosing
    Returns:
        jpeg_data_placeholder: tensor, we should give it result of tf.read('target_image.jpeg')
        feature_vector: our blackbox output
    '''
    module = hub.Module(module_path)
    input_file, resized_image = add_image_deocding(module)
    feature_vector = module(resized_image)
    return input_file, feature_vector

data_set = './Data/memes'
feature_vectors = './Data/feature_vectors'
input_tensor, feature_vector = get_feature_extractor()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for image in os.listdir(data_set):
        try:
            feature_vector_calculated = sess.run(feature_vector, feed_dict={input_tensor:os.path.join(data_set, image)})
            np.savetxt(os.path.join(feature_vectors, image[:-3]+'txt'), feature_vector_calculated)
        except Exception:
            print(image)