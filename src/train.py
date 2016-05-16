from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from sklearn.utils import shuffle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import deepModel
import mrcReader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_epochs', 300,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

batch_size = deepModel.batch_size
num_class = deepModel.num_class
image_size = deepModel.image_size
train_size = deepModel.num_examples_train
eval_size = deepModel.num_examples_eval

eval_frequency = train_size // batch_size

best_error_rate = 100
patience = 10000

model_save_file = '../result/cnn_pickleModel1.ckpt'


def error_rate(prediction, label):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (100.0 * np.sum(np.argmax(prediction, 1) == label) / prediction.shape[0])

def train():
    global best_error_rate
    global patience
    train_data, train_label, eval_data, eval_label = mrcReader.loadInputData()
    train_data, train_label = shuffle(train_data, train_label)
    eval_data, eval_label = shuffle(eval_data, eval_label)

    train_data_node = tf.placeholder(
                    tf.float32,
                    shape=(batch_size, image_size, image_size, 1))
    train_label_node = tf.placeholder(tf.int64, shape=(batch_size,))
    eval_data_node = tf.placeholder(
                    tf.float32,
                    shape=(batch_size, image_size, image_size, 1))

#        deepModel.variable_init()
    global_step = tf.Variable(0, trainable=False)
    logits = deepModel.inference(train_data_node, train=True)
    loss, pred = deepModel.loss(logits, train_label_node)
    opt, lr = deepModel.train(loss, global_step)

    eval_logits = deepModel.inference(eval_data_node, train=False)
    eval_pred = deepModel.loss(eval_logits, None) 

    saver = tf.train.Saver(tf.all_variables())

    init = tf.initialize_all_variables()
    
    def evaluation(data, sess):
        size = eval_size
        predictions = np.ndarray(shape=(size, num_class), dtype=np.float32)
        for begin in xrange(0, size, batch_size):
            end = begin + batch_size
            if end <= size:
                batch_data = data[begin:end, ...]
                predictions[begin:end, :] = sess.run(
                    eval_pred,
                    feed_dict={eval_data_node: batch_data})
            else:
                batch_data = data[-batch_size:, ...]
                batch_predictions = sess.run(
                    eval_pred,
                    feed_dict={eval_data_node: batch_data})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    start_time = time.time()
#    print((FLAGS.max_epochs * train_size) // batch_size)
    with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as sess:
        sess.run(init)
        for step in xrange(int(FLAGS.max_epochs * train_size) // batch_size):
            offset =  (step * batch_size) % (train_size - batch_size)
            batch_data = train_data[offset:(offset+batch_size),...]
            batch_label = train_label[offset:(offset+batch_size)]
#            print(batch_label)
            feed_dict = {train_data_node: batch_data, train_label_node: batch_label} 
            _, loss_value, prediction = sess.run(
                                        [opt, loss, pred], feed_dict=feed_dict)
#            print(step, k)
#            print('----------------------------------')
            if step % eval_frequency == 0:
                stop_time = time.time() - start_time
                start_time = time.time()
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size,
                                                1000 * stop_time / eval_frequency)) 
                print('train loss: %.3f' % loss_value) 
                print('train error: %.2f%%' % error_rate(prediction, batch_label))         
                eval_prediction = evaluation(eval_data, sess)
                eval_error_rate = error_rate(eval_prediction, eval_label)
                print('valid error: %.2f%%' % eval_error_rate)
                
                if(eval_error_rate < best_error_rate):
                    if(eval_error_rate < best_error_rate * 0.95):
                        if(patience<step *2):
                            patience = patience *2
                    best_error_rate = eval_error_rate
            if step >= patience:
                saver.save(sess, model_save_file, global_step = step)
                break
            if (step +1) == (FLAGS.max_epochs * train_size) // batch_size:
                saver.save(sess, model_save_file, global_step = step)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
