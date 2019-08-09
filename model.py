#! /usr/bin/python3
import tensorflow as tf
import os
import json
import math
import itertools

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
        first_layer  = tf.layers.dense(encoded_layer, units = 1024, activation=tf.nn.relu)
        second_layer  = tf.layers.dense(first_layer , units = 512,  activation=tf.nn.relu)
        third_layer  = tf.layers.dense(second_layer , units = 256,  activation=tf.nn.relu)
        output  = tf.layers.dense(third_layer, units = 1 )
        return image_encoded, sentence_encoded, output

    def calculate_loss(self, output, truth):
        return tf.reduce_mean(tf.square(output - truth))
    
    def get_optimizer(self, loss, learning_rate = 0.01):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    def train(self, number_of_steps):
        data = Data()
        self.sess.run(tf.global_variables_initializer())
        for _ in range(number_of_steps):
            for batch in data.batches():
                _, l = self.sess.run((self.optimizer, self.loss), \
                    feed_dict={self.image_encoded: batch['image'], self.sentence_encoded: batch['sentence'], self.scores: batch['scores']})
                print(l)
class Data:
    def __init__(self):
        self.image_encoding_dir = './Data/image_feature_vectors'
        self.image_encodings = {}
        for file in os.listdir(self.image_encoding_dir):
            with open(os.path.join(self.image_encoding_dir,file), 'r') as f:
                vector = [float(x) for x in f.read().split(' ')]
            self.image_encodings[file[:-4]] = vector
        
        self.sentence_encoding_dir = './Data/sentence_feature_vectors'
        self.sentence_encodings = {}
        for file in os.listdir(self.sentence_encoding_dir):
            with open(os.path.join(self.sentence_encoding_dir,file), 'r') as f:
                vector = [float(x) for x in f.read().split(' ')]
            self.sentence_encodings[file[:-4]] = vector
        
        self.json_file = './Data/db.json'
        json_file_loaded = json.load(open(self.json_file))

        valid_keys = self.image_encodings.keys() & self.sentence_encodings.keys()

        self.scores = {v['id'] : math.log(v['ups']) for v in json_file_loaded['_default'].values() if v['id'] in valid_keys}
        self.data = [(self.image_encodings[x],self.sentence_encodings[x],self.scores[x]) for x in valid_keys if x in self.image_encodings and x in self.sentence_encodings and x in self.scores]
    def batches(self, number_of_sampels = 100):
        batch = {'image':[],'sentence':[],'scores':[]}
        for i in self.data:
            batch['image'].append(i[0])
            batch['sentence'].append(i[1])
            batch['scores'].append([i[2]])
            if len(batch) == number_of_sampels:
                yield batch
                batch = {'image':[],'sentence':[],'scores':[]}
        yield batch

model = Model()
model.train(100)