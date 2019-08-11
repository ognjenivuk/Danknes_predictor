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
import magic

def _test_train_split(data: list, split = 0.8):
    data_copy = data.copy()
    random.shuffle(data_copy)
    split_index = int(len(data_copy) * split)
    return data_copy[:split_index], data_copy[split_index:]

def corr(x, y):
    return np.corrcoef(x.flatten(), y.flatten())[1, 0]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Model:
    def __init__(self, use_text = True, reg_param = 0.01, dro_param = 0.3, use_image = True):
        self.sess = tf.Session()
        self.keep_prob = tf.placeholder_with_default(dro_param or 1.0, (), name='keep_prob')
        self.image_encoded_1, self.image_encoded_2, self.sentence_encoded_1, self.sentence_encoded_2, self.output = self.load_model_layers(use_text, use_image)
        self.scores = tf.placeholder(tf.float32, shape=(None, 1), name='scores')
        self.loss, self.crossentropy = self.calculate_loss(self.output, self.scores, reg_param)
        self.optimizer = self.get_optimizer(self.loss)

    def _create_fc_layers(self, sizes, input, kernel_regularizer, activation):
        layer = tf.layers.dense(input, units=sizes[0], activation = activation, kernel_regularizer = kernel_regularizer)
        dp = tf.nn.dropout(layer, keep_prob=self.keep_prob)

        for i in range(1, len(sizes)):
            layer = tf.layers.dense(dp, units=sizes[i], activation = activation, kernel_regularizer = kernel_regularizer)
            dp = tf.nn.dropout(layer, keep_prob=self.keep_prob)
                    
        return dp
        

    def load_model_layers(self, use_text, use_image):
        regulizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0)
        
        image_encoded_1 = tf.placeholder(tf.float32, shape = (None, 1280), name='image_encoded_1')
        image_encoded_2 = tf.placeholder(tf.float32, shape = (None, 1280), name='image_encoded_2')
        sentence_encoded_1 = tf.placeholder(tf.float32, shape = (None, 512), name='sentence_encoded_1')
        sentence_encoded_2 = tf.placeholder(tf.float32, shape = (None, 512), name='sentence_encoded_2')

        l = []
        if use_text:
            l.append(sentence_encoded_1)
            l.append(sentence_encoded_2)

        if use_image:
            l.append(image_encoded_1)
            l.append(image_encoded_2)

        all_encoded = tf.concat(l, axis = 1)
        last_fc_layer = self._create_fc_layers(
            [512], 
            all_encoded,
            kernel_regularizer=regulizer, 
            activation=tf.nn.relu
        )
        output = tf.layers.dense(last_fc_layer, units=1, name = 'output', activation=tf.nn.sigmoid)
        return image_encoded_1, image_encoded_2, sentence_encoded_1, sentence_encoded_2, output

    def calculate_loss(self, output, truth, beta):
        cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=truth,
            logits=output
        ))
        regularization_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return cross_entropy + beta*regularization_loss, cross_entropy
    
    def get_optimizer(self, loss, learning_rate = 0.01):
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        

    def _run_batch(self, batch, fetches, no_dropout=False):
        '''runs a batch and returns the loss'''
        feed_dict = {
            self.image_encoded_1: batch['image'][:, 0], 
            self.image_encoded_2: batch['image'][:, 1], 
            self.sentence_encoded_1: batch['sentence'][:, 0], 
            self.sentence_encoded_2: batch['sentence'][:, 1], 
            self.scores: batch['scores'].reshape((-1, 1))
        }

        if no_dropout:
            feed_dict[self.keep_prob] = 1
        
        return self.sess.run(
            fetches, 
            feed_dict=feed_dict
        )

            
    def _plot_train_model(self):
        for batch in self.data.train_batches(number_of_samples=-1):
            output = self._run_batch(batch, self.output)
            truth = sigmoid(batch['scores'][:,0] - batch['scores'][:,1])

        plt.plot(truth, truth, 'r-')
        plt.plot(output, truth, '.')
        plt.legend(['ground_truth', 'output'])
        plt.xlabel('output')
        plt.ylabel('truth')
        plt.title(f'output on training set\n{output[:5]}')
        plt.show()

    def train(self, number_of_steps, save_name):
        self.data = Data()
        self.sess.run(tf.global_variables_initializer())

        if save_name:
            saver = tf.train.Saver()
            save_path = './trained_models/model_' + save_name + str(datetime.now()) +'.cpkt'
        
        training_losses = []
        validation_losses = []

        # magic.init_loss()

        try:
            for cur_step in range(number_of_steps):

                this_epoch_losses = []

                for batch in self.data.train_batches(number_of_samples=1000):
                    _, l, cross_entropy = self._run_batch(batch, (self.optimizer, self.loss, self.crossentropy)) 
                    this_epoch_losses.append(cross_entropy)
                training_losses.append(sum(this_epoch_losses) / len(this_epoch_losses))
                
                for batch in self.data.validation_batches():
                    validation_loss, validation_cross_entropy, validation_output = self._run_batch(batch, (self.loss, self.crossentropy, self.output), no_dropout=True)
                    validation_truth = np.array(batch['scores'])
                validation_losses.append(validation_loss)
                progress = '=' * math.ceil(cur_step / number_of_steps * 40) + '>'
                progress = progress.ljust(40)
                progress = progress[:40]

                # magic.add_vals(training_losses[-1], validation_cross_entropy)

                corr_coef = 0#corr(validation_output, validation_truth)
                output_std = 0#validation_output.std()
                intput_std = 0#validation_truth.std()

                print('[', progress, ']', 
                    f' [\x1b[31mtrain cross_entropy : {training_losses[-1]:.4f}\x1b[0m' 
                    f', \x1b[32mval cross_entropy : {validation_cross_entropy:.4f}\x1b[0m, stat : {intput_std:.2f} {corr_coef:.2f} {output_std:.2f}] {cur_step} / {number_of_steps}                       ', end='\r', sep='')
                    
            print()
        except KeyboardInterrupt:
            pass

        self._plot_train_model()

        if save_name:
            saver.save(self.sess, save_path)
            print()
            print('model saved at ' + save_path)
        
        return training_losses, validation_losses

    def _are_you_dank(self, score):
        return (np.ones_like(score, dtype=np.int32) * 3) - (score < 2.7) - (score < 4.7) - (score < 7.8)

    def test_classy(self):
        for batch in self.data.validation_batches():
            output, cross_entropy = self._run_batch(batch, (self.output, self.crossentropy), no_dropout=True)
            truth = np.array(batch['scores'])

        plt.plot(truth, truth, 'r-')
        plt.plot(output, truth, '.')
        plt.legend(['ground_truth', 'output'])
        plt.xlabel('output')
        plt.ylabel('truth')
        correlation = np.corrcoef(truth.flatten(), output.flatten())[1,0]
        plt.title(f'output on validation set \n  cross_entropy on validation set : {cross_entropy} \n rho = {correlation}')
        plt.show()

        # true_classes = self._are_you_dank(truth).astype(np.int32).flatten()
        # output_classes = self._are_you_dank(output).astype(np.int32).flatten()
        # plot_confusion_matrix(
        #     true_classes, 
        #     output_classes, 
        #     np.array(['bad', 'meh', 'less meh', 'dank']))
        # plt.title('confusion matrix on test set')
        # plt.show()

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

    def _get_batch(self, data, number_of_samples):
        batch = {'image':[],'sentence':[],'scores':[]}

        half = len(data) // 2
        data_1, data_2 = data[:half], data[half:]

        for d1, d2 in zip(data_1, data_2):
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


SAVE_NAME_WO_TEXT = 'wo_text'
SAVE_NAME_ALL = 'from_hub_modules'

def main():
    seed = 6
    random.seed(seed)
    tf.random.set_random_seed(seed)

    # data = Data()

    # x = next(data._get_batch(data.test_data, -1))

    # x = x['scores'][:,0] - x['scores'][:,1]
    # plt.hist(x, bins=50)
    # plt.show() 

    # return

    # scores_train = [x[2] for x in data.train_data]
    # scores_test = [x[2] for x in data.test_data]
    
    # plt.hist(scores_train, 100)
    # plt.hist(scores_test, 100)
    # plt.legend(['train scores', 'test scores'])
    # plt.title(f'train and test scores histogram')
    # plt.show()

    model = Model(reg_param=0, dro_param=None)
    train_losses, validation_losses = model.train(1000, save_name=None)
    plt.plot(train_losses)
    plt.plot(validation_losses)
    plt.legend(['train', 'validation'])
    plt.title('loss on train/validation')
    plt.show()

    # test_loss, test_output = model.test()
    # plt.hist2d(scores_test, test_output.flatten(), bins = 50)
    # plt.title(f'2d histogram of test output/test gorund truth \n test loss is {test_loss}')
    # plt.show()
    #
    # print(f'test loss {test_loss}')

    # model.test_classy()
        
if __name__ == '__main__':
    main()