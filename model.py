#! /usr/bin/python3
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

def _test_train_split(data: list, split = 0.8):
    data_copy = data.copy()
    random.shuffle(data_copy)
    split_index = int(len(data_copy) * split)
    return data_copy[:split_index], data_copy[split_index:]

class Model:
    def __init__(self, use_text = True):
        self.image_encoded, self.sentence_encoded, self.output = self.load_model_layers(use_text)
        self.scores = tf.placeholder(tf.float32, shape=(None, 1), name='scores')
        self.loss = self.calculate_loss(self.output, self.scores)
        self.optimizer = self.get_optimizer(self.loss)
        self.sess = tf.Session()

    def load_model_layers(self, use_text = True):
        image_encoded = tf.placeholder(tf.float32, shape = (None, 2048), name='image_encoded')
        sentence_encoded = tf.placeholder(tf.float32, shape = (None, 512), name='sentence_encoded')
        both_encoded = tf.concat([image_encoded, sentence_encoded], axis = 1)
        first_layer  = tf.layers.dense(both_encoded if use_text else image_encoded, units = 512,  activation=tf.nn.relu)
        tf.layers.dropout(first_layer, 0.3)
        second_layer  = tf.layers.dense(first_layer , units = 256,  activation=tf.nn.relu)
        tf.layers.dropout(second_layer, 0.3)
        output  = tf.layers.dense(second_layer, units = 1, name = 'output' )
        return image_encoded, sentence_encoded, output

    def calculate_loss(self, output, truth):
        return tf.reduce_mean(tf.square(output - truth))
    
    def get_optimizer(self, loss, learning_rate = 0.01):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    def _run_batch(self, batch, fetches):
        '''runs a batch and returns the loss'''
        return self.sess.run(
            fetches, 
            feed_dict={self.image_encoded: batch['image'], self.sentence_encoded: batch['sentence'], self.scores: batch['scores']})
            
    def _plot_train_model(self):
        for batch in self.data.train_batches(number_of_samples=-1):
            output = self._run_batch(batch, self.output)
            truth = np.array(batch['scores'])

        plt.plot(output, output, 'r-')
        plt.plot(output, truth, '.')
        plt.show()

    def train(self, number_of_steps, save_name):
        self.data = Data()
        self.sess.run(tf.global_variables_initializer())

        if save_name:
            saver = tf.train.Saver()
            save_path = './trained_models/model_' + save_name + str(datetime.now()) +'.cpkt'
        
        training_losses = []
        validation_losses = []

        try:
            for cur_step in range(number_of_steps):

                this_epoch_losses = []

                for batch in self.data.train_batches(number_of_samples=1000):
                    _, l = self._run_batch(batch, (self.optimizer, self.loss)) 
                    this_epoch_losses.append(l)

                training_losses.append(sum(this_epoch_losses) / len(this_epoch_losses))
                
                for batch in self.data.validation_batches():
                    validation_loss = self._run_batch(batch, self.loss)
                validation_losses.append(validation_loss)
                progress = '=' * math.ceil(cur_step / number_of_steps * 80) + '>'
                progress = progress.ljust(80)
                progress = progress[:80]
                print('[', progress, ']', 
                    f' [\x1b[31mtrain loss : {training_losses[-1]:.4f}\x1b[0m' 
                    f', \x1b[32mval loss : {validation_loss:.4f}\x1b[0m]                                  ', end='\r', sep='')
                    
            print()
        except KeyboardInterrupt:
            pass

        self._plot_train_model()

        if save_name:
            saver.save(self.sess, save_path)
            print('model saved at ' + save_path)
        
        return training_losses, validation_losses
    
    def test(self):
        for batch in self.data.test_batches(number_of_samples=len(self.data.test_data)):
            test_loss = self.sess.run(self.loss, \
                feed_dict={self.image_encoded: batch['image'], self.sentence_encoded: batch['sentence'], self.scores: batch['scores']})
        return test_loss

    def _are_you_dank(self, score):
        return (np.ones_like(score, dtype=np.int32) * 3) - (score < 7.3) - (score < 8.9) - (score < 10.6)

    def test_classy(self):
        for batch in self.data.test_batches():
            output = self._run_batch(batch, self.output)
            truth = np.array(batch['scores'])

        plt.plot(output, output, 'r-')
        plt.plot(output, truth, '.')
        plt.title('this is loss: ' + str(np.mean(np.square(output - truth))))
        plt.show()

        true_classes = self._are_you_dank(truth).astype(np.int32).flatten()
        output_classes = self._are_you_dank(output).astype(np.int32).flatten()

        print(true_classes)
        print(output_classes)
        
        plot_confusion_matrix(
            true_classes, 
            output_classes, 
            np.array([0, 1, 2, 3]))

        plt.show()

class Data:
    def __init__(self):
        image_encoding_dir = './Data/image_feature_vectors'
        self.image_encodings = read_vectors_from_files(image_encoding_dir)
        
        sentence_encoding_dir = './Data/sentence_feature_vectors'
        self.sentence_encodings = read_vectors_from_files(sentence_encoding_dir)
        
        self.scores = load_scores('./Data/db.json')

        self.data = [
            (self.image_encodings[x],self.sentence_encodings[x],self.scores[x])
            for x in self.image_encodings.keys()
            if x in self.sentence_encodings and x in self.scores
        ]

        self.train_data, self.test_data = _test_train_split(self.data, 0.2)
        self.train_data, self.validation_data = _test_train_split(self.train_data, 0.75)

    def train_batches(self, number_of_samples = 100):
        return self._get_batch(self.train_data, number_of_samples)

    def test_batches(self, number_of_samples = -1):
        return self._get_batch(self.test_data, number_of_samples)

    def validation_batches(self, number_of_samples = -1):
        return self._get_batch(self.validation_data, number_of_samples)

    def _get_batch(self, data, number_of_samples):
        batch = {'image':[],'sentence':[],'scores':[]}
        for i in data:
            batch['image'].append(i[0])
            batch['sentence'].append(i[1])
            batch['scores'].append([i[2]])
            if len(batch['image']) == number_of_samples:
                yield batch
                batch = {'image':[],'sentence':[],'scores':[]}
                
        if batch['image']:
            yield batch


SAVE_NAME_WO_TEXT = 'wo_text'
SAVE_NAME_ALL = 'from_hub_modules'

def main():
    seed = 5
    random.seed(seed)
    tf.random.set_random_seed(seed)

    model = Model(use_text=True)
    train_losses, validation_losses = model.train(2000, save_name=SAVE_NAME_ALL)
    plt.plot(np.clip(train_losses, 0, 70))
    plt.plot(np.clip(validation_losses, 0, 70))
    plt.legend(['train', 'validation'])
    plt.title('is this loss')
    plt.show()
    
    test_loss = model.test()
    print(f'test loss {test_loss}')

    model.test_classy()
    
if __name__ == '__main__':
    main()