#!/usr/bin/python3
import tensorflow as tf
import os
import json
import math
import random
import itertools
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from data import read_vectors_from_files, load_scores, plot_confusion_matrix
import magic
from scipy.stats import binom_test

def _test_train_split(data, split):
    data2 = data.copy()
    random.shuffle(data2)
    split_index = int(len(data2) * split)
    return data2[:split_index], data2[split_index:]

class Data:
    def __init__(self):
        image_encoding_dir = './Data/new_data/image_feature_vectors'
        self.image_encodings = read_vectors_from_files(image_encoding_dir)

        sentence_encoding_dir = './Data/new_data/sentence_feature_vectors'
        self.sentence_encodings = read_vectors_from_files(sentence_encoding_dir)
        
        self.scores = load_scores('./Data/new_data/db.json')

        self.data = [
            (self.image_encodings[x],self.sentence_encodings[x],self.scores[x])
            for x in self.image_encodings.keys()
            if x in self.sentence_encodings and x in self.scores
        ]

        self.train_data, self.test_data = _test_train_split(self.data, 0.8)
        self.train_data, self.validation_data = _test_train_split(self.train_data, 0.75)

    def train_batches(self, number_of_samples = 100):
        random.shuffle(self.train_data)
        return self._get_batch(self.train_data, number_of_samples)

    def test_batches(self, number_of_samples = -1):
        return self._get_batch(self.test_data, number_of_samples)

    def validation_batches(self, number_of_samples = -1):
        random.shuffle(self.test_data)
        return self._get_batch(self.validation_data, number_of_samples)

    def _make_pairs(self, data):
        data2 = data.copy()
        data2.reverse()
        return zip(data, data2)

    def _get_batch(self, data, number_of_samples):
        batch = {'image':[],'sentence':[],'scores':[]}

        for d1, d2 in self._make_pairs(data):
            batch['image'].append((d1[0], d2[0]))
            batch['sentence'].append((d1[1], d2[1]))
            batch['scores'].append(d1[2] > d2[2])
            if len(batch['image']) == number_of_samples:
                yield {
                    'image': np.array(batch['image']),
                    'sentence': np.array(batch['sentence']),
                    'scores': np.array(batch['scores']),
                }
                batch = {'image':[],'sentence':[],'scores':[]}
                
        if batch['image']:
            yield {
                'image': np.array(batch['image']),
                'sentence': np.array(batch['sentence']),
                'scores': np.array(batch['scores']),
            }

class Model:
    def __init__(self, dro_param = None, use_text = True, beta = 0, learning_rate = 0.01):
        self.sess = tf.Session()
        self.keep_prob = tf.placeholder_with_default(dro_param or 1.0, (), name='keep_prob')

        self.load_network_layers(use_text=use_text)
        
        self.truth = tf.placeholder(tf.float32, shape = (None,1))
        self.loss, self.cross_entropy = self.calculate_loss(self.output, self.truth, beta)
        self.optimizer = self.get_optimizer(self.loss, learning_rate)

    def _make_fc_layers(self, input, units, activation, keep_prob, kernel_regularizer):

        def make_dense(inp, size):
            layer = tf.layers.dense(inp, units=size, 
                activation=activation,
                #kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.random_normal_initializer(0.0, 1.0)
            )
            return layer

        layer = make_dense(input, units[0])
        for i in range(1, len(units)):
            layer = make_dense(layer, units[i])
        return layer

    def save_grpah(self):
        writer = tf.summary.FileWriter('/tmp/logdir', graph=self.sess.graph)
        writer.close()
    
    def load_network_layers(self, use_text):
        regulizer = tf.contrib.layers.l2_regularizer(1.0)
        self.image_encoded_1 = tf.placeholder(tf.float32, shape = (None, 1280), name='image_encoded_1')
        self.image_encoded_2 = tf.placeholder(tf.float32, shape = (None, 1280), name='image_encoded_2')
        self.sentence_encoded_1 = tf.placeholder(tf.float32, shape = (None, 512), name='sentence_encoded_1')
        self.sentence_encoded_2 = tf.placeholder(tf.float32, shape = (None, 512), name='sentence_encoded_2')

        input_list = [self.image_encoded_1, self.sentence_encoded_1, self.image_encoded_2, self.sentence_encoded_2] if use_text else\
            [self.image_encoded_1, self.image_encoded_2]
            
        input_layer = tf.concat(input_list, axis = 1)
        
        last_fc_layer = self._make_fc_layers(input_layer, [1024, 1024, 1024], activation=tf.nn.relu, keep_prob = self.keep_prob, kernel_regularizer=regulizer)
        self.output = tf.layers.dense(last_fc_layer, units = 1, activation = None, kernel_initializer=tf.random_normal_initializer(0.0, 1.0))
        self.output_for_inference = tf.nn.sigmoid(self.output)
        

    def calculate_loss(self, logits, labels, beta):
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels), axis=1)
        return cross_entropy + beta*regularization_loss, cross_entropy
    
    def get_optimizer(self, loss, learning_rate):
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    def _run_batch(self, fetch, batch):
        output = self.sess.run(fetch, feed_dict={
            self.image_encoded_1: batch['image'][:, 0],
            self.image_encoded_2: batch['image'][:, 1],
            self.sentence_encoded_1: batch['sentence'][:, 0],
            self.sentence_encoded_2: batch['sentence'][:, 1],
            self.truth: batch['scores'].reshape((-1, 1)).astype(np.float32)
        })

        return output

    def train(self, epochs):

        data = Data()
        self.sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        
        training_losses = []
        validation_losses = []
        try:
        
            for epoch in range(epochs):
                
                cross_ents = []
                for batch in data.train_batches(1000):
                    _, ce = self._run_batch((self.optimizer, self.cross_entropy), batch)
                    cross_ents.append(ce)

                training_loss = np.average(cross_ents[0])
                training_losses.append(training_loss)

                for batch in data.validation_batches(-1):
                    ce, val_output, val_truth = self._run_batch([ self.cross_entropy, self.output_for_inference, self.truth ], batch)
                    val_output = np.where(val_output > 0.5, 1, 0).flatten().astype(np.int32)
                    val_truth = np.where(val_truth > 0.5, 1, 0).flatten().astype(np.int32)

                validation_losses.append(np.average(ce))
                validation_acc = np.mean(val_output == val_truth) * 100

                print(f'| epoch {epoch} | trn ce {training_loss:.4f} | val ce {validation_losses[-1]:.4f} | val acc {validation_acc:.2f}                  ', end='\r')
        except KeyboardInterrupt:
            pass
        
        print()

        plt.plot(validation_losses)
        plt.plot(training_losses)
        plt.legend(['validation', 'training'])
        plt.title('cross-entropy')
        plt.show()

        self._plot_results(data.train_batches(-1), 'train')
        self._plot_results(data.validation_batches(-1), 'validation')

        
        save_name = input('model save name [default: dont save]: ')

        if save_name:
            save_path = './trained_models/model_' + save_name + '_' + str(datetime.now()) +'.cpkt'
            saver.save(self.sess, save_path)
            print()
            print('model saved at ' + save_path)
            
    def _plot_results(self, batches, name):

        output = []
        truth = []

        for batch in batches:
            out, tru = self._run_batch([ self.output_for_inference, self.truth ], batch)
            output.append(out)
            truth.append(tru)

        output = np.array(output).flatten()
        truth = np.array(truth).flatten()

        plt.hist(output, range=(0.0, 1.0), bins=50)
        plt.title('output distribution on ' + name)
        plt.show()

        plt.hist(truth, bins=50, range=(0.0, 1.0))
        plt.title('truth distribution on ' + name)
        plt.show()

        def step(x, point=0.5):
            return np.where(x > point, 1, 0).astype(np.int32)

        truth = step(truth)
        output = step(output)
        
        plot_confusion_matrix(truth, output, np.array([0, 1]), 
            title='confusion matrix for ' + name
        )
        plt.show()

        accuracy = np.average(truth == output)
        stat_sig = binom_test(np.count_nonzero(truth==output), n=len(truth))
        print()
        print(f' ==== {name} ==== ')
        print(f'accuracy = {accuracy}')
        print(f'       p = {stat_sig}')    
                

def main():
    #beta = regularization param
    #dro_param = keep_prbo in dropout
    model = Model(dro_param = 1.0, beta = 0, learning_rate = 0.01)
    
    model.save_grpah()

    model.train(1000)
    
if __name__ == "__main__":
    main()