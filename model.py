#! /usr/bin/python3
import tensorflow as tf
import os
import json
import math
import random
import itertools

def _test_train_split(data: list, split = 0.8):
    data_copy = data.copy()
    random.shuffle(data_copy)
    split_index = int(len(data_copy) * split)
    return data_copy[:split_index], data_copy[split_index:]

class Model:
    def __init__(self):
        self.image_encoded, self.sentence_encoded, self.output = self.load_model_layers()
        self.scores = tf.placeholder(tf.float32, shape=(None, 1))
        self.loss = self.calculate_loss(self.output, self.scores)
        self.optimizer = self.get_optimizer(self.loss)
        self.sess = tf.Session()

    def load_model_layers(self):
        image_encoded = tf.placeholder(tf.float32, shape = (None, 2048))
        sentence_encoded = tf.placeholder(tf.float32, shape = (None, 512))
        encoded_layer = tf.concat([image_encoded, sentence_encoded], axis = 1)
        first_layer  = tf.layers.dense(encoded_layer , units = 512,  activation=tf.nn.relu)
        second_layer  = tf.layers.dense(first_layer , units = 256,  activation=tf.nn.relu)
        output  = tf.layers.dense(second_layer, units = 1 )
        return image_encoded, sentence_encoded, output

    def calculate_loss(self, output, truth):
        return tf.reduce_mean(tf.square(output - truth))
    
    def get_optimizer(self, loss, learning_rate = 0.01):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    def train(self, number_of_steps):
        self.data = Data()
        self.sess.run(tf.global_variables_initializer())
        for _ in range(number_of_steps):
            for batch in self.data.batches(is_training = True):
                _, l = self.sess.run((self.optimizer, self.loss), \
                    feed_dict={self.image_encoded: batch['image'], self.sentence_encoded: batch['sentence'], self.scores: batch['scores']})
                print(l)
    def test(self):
        for batch in self.data.batches(is_training = False, number_of_sampels=len(self.data.test_data)):
            test_loss = self.sess.run(self.loss, \
                feed_dict={self.image_encoded: batch['image'], self.sentence_encoded: batch['sentence'], self.scores: batch['scores']})
        return test_loss

def _read_vectors_from_files(file_dir):
    encodings = {}
    for vector_file in os.listdir(file_dir):
        with open(os.path.join(file_dir, vector_file), 'r') as f:
            encodings[os.path.splitext(vector_file)[0]] = [float(x) for x in f.read().split(' ')]

    return encodings


class Data:
    def __init__(self):
        image_encoding_dir = './Data/image_feature_vectors'
        self.image_encodings = _read_vectors_from_files(image_encoding_dir)
        
        sentence_encoding_dir = './Data/sentence_feature_vectors'
        self.sentence_encodings = _read_vectors_from_files(sentence_encoding_dir)
        
        json_file = './Data/db.json'
        with open(json_file) as f:
            json_file_loaded = json.load(f)

        valid_keys = self.image_encodings.keys() & self.sentence_encodings.keys()

        self.scores = {v['id'] : math.log(v['ups']) for v in json_file_loaded['_default'].values() if v['id'] in valid_keys}
        self.data = [
            (self.image_encodings[x],self.sentence_encodings[x],self.scores[x])
            for x in valid_keys
            if x in self.image_encodings and x in self.sentence_encodings and x in self.scores
        ]

        self.train_data, self.test_data = _test_train_split(self.data)


    def batches(self, is_training ,number_of_sampels = 100):
        batch = {'image':[],'sentence':[],'scores':[]}
        current_data = self.train_data if is_training else self.test_data
        for i in current_data:
            batch['image'].append(i[0])
            batch['sentence'].append(i[1])
            batch['scores'].append([i[2]])
            if len(batch) == number_of_sampels:
                yield batch
                batch = {'image':[],'sentence':[],'scores':[]}
        if batch['image']:
            yield batch

model = Model()
model.train(10)
test_loss = model.test()
print(f'test loss {test_loss}')