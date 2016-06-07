"""
build pick model
 # training, input images and labels
 data, label = disored_inputs()

 # evaluation, input images

 # inference, inference the model from input to prediction
 logits = inference(inputs)

 # loss, compute the loss function 
 loss = loss(logits, label)

 #create a graph to train step one by one
 train_op = train(loss, global_step)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

""" create variable with weight decay    
 """

SEED = 6543
def _variable_with_weight_decay(name, shape, stddev, wd):
    
    var = tf.get_variable(name, shape, 
                        initializer = tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
    return var

class DeepModel ():
    def __init__(self, image_size, num_class, batch_size,  num_train=0, num_eval=0):
# image data constants information
        self.image_size = image_size
        self.num_class = num_class
        self.num_examples_train = num_train
        self.num_examples_eval = num_eval

        # model constants parameter
        self.learning_rate = 0.01
        self.learning_rate_decay_factor = 0.1
        self.momentum = 0.9
        self.num_epochs_per_decay = 400
        self.batch_size = batch_size

        #def variable_init():
        self.kernel1 = _variable_with_weight_decay('weights1', shape=[5, 5, 1, 8],
                                            stddev=0.05, wd = 0.0)
        self.biases1 = tf.get_variable('biases1', [8], initializer=tf.constant_initializer(0.0))
        self.kernel2 = _variable_with_weight_decay('weights2', shape=[5, 5, 8, 16],
                                            stddev=0.05, wd = 0.0)
        self.biases2 = tf.get_variable('biases2', [16], initializer=tf.constant_initializer(0.0))
        self.kernel3 = _variable_with_weight_decay('weights3', shape=[3, 3, 16, 32],
                                            stddev=0.05, wd = 0.0)
        self.biases3 = tf.get_variable('biases3', [32], initializer=tf.constant_initializer(0.0))
        self.kernel4 = _variable_with_weight_decay('weights4', shape=[2, 2, 32, 64],
                                            stddev=0.05, wd = 0.0)
        self.biases4 = tf.get_variable('biases4', [64], initializer=tf.constant_initializer(0.0))

        dim = 64*2*2
        self.weights_fc1 = _variable_with_weight_decay('weightsf1', shape=[dim, 128],
                                            stddev=0.05, wd=0.0005)
        self.biases_fc1 = tf.get_variable('biasesf1', [128], initializer=tf.constant_initializer(0.0))
        self.weights_fc2 = _variable_with_weight_decay('weightsf2', shape=[128, num_class],
                                            stddev=0.05, wd=0.0005)
        self.biases_fc2 = tf.get_variable('biasesf2', [num_class], initializer=tf.constant_initializer(0.0))


    """ build cnn model, 
        input : data
        return : predictions
    """
    def inference(self, data, train=True):
        print("hhh")
        conv1 = tf.nn.conv2d(data, 
                            self.kernel1, 
                            strides=[1, 1, 1, 1],
                            padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.biases1))

        pool1 = tf.nn.max_pool(relu1, 
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

        conv2 = tf.nn.conv2d(pool1, 
                            self.kernel2, 
                            strides=[1, 1, 1, 1],
                            padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.biases2))

        pool2 = tf.nn.max_pool(relu2, 
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

        conv3 = tf.nn.conv2d(pool2, 
                            self.kernel3, 
                            strides=[1, 1, 1, 1],
                            padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.biases3))

        pool3 = tf.nn.max_pool(relu3, 
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

        conv4 = tf.nn.conv2d(pool3, 
                            self.kernel4, 
                            strides=[1, 1, 1, 1],
                            padding='VALID')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, self.biases4))

        pool4 = tf.nn.max_pool(relu4, 
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

        hidden = tf.reshape(pool4, [self.batch_size, -1])
        print(hidden.get_shape())
        if train:
            hidden = tf.nn.dropout(hidden, 0.3, seed=SEED)

        fc1 = tf.nn.relu(tf.matmul(hidden, self.weights_fc1) + self.biases_fc1)

        sotfmax = tf.add(tf.matmul(fc1, self.weights_fc2), self.biases_fc2)

        return (sotfmax)

    """
     compute loss with prediction and label, also will acount for L2loss
     # input : prediction, label
     # output : loss
    """
    def loss(self, logits, label):

        prediction = tf.nn.softmax(logits)
        print(prediction.get_shape())
        if label is not None:
            labels = tf.cast(label, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits, label, name = 'cross_entropy_all')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)
            all_loss = tf.add_n(tf.get_collection('losses'), name='all_loss')
            return (all_loss, prediction)
        else:
            return prediction

    def train(self, all_loss, global_step):
        
        num_batches_per_epoch = self.num_examples_train / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        lr = tf.train.exponential_decay(self.learning_rate,
                                        global_step,
                                        decay_steps,
                                        self.learning_rate_decay_factor, staircase=True)
        optimizer = tf.train.MomentumOptimizer(lr, self.momentum).minimize(all_loss, 
                                                    global_step = global_step)

    #    grads = optimizer.compute_gradients(all_loss)
    #    apply_gradient_op = optimizer.apply_gradients(grads, global_step = global_step)

    #    with tf.control_dependencies([optimizer]):
    #        train_op = tf.no_op(name='train')

        return (optimizer, lr)
