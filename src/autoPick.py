from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import deepModel
import mrcReader

batch_size = deepModel.batch_size
image_size = deepModel.image_size
num_class = deepModel.num_class
test_size = 218

checkpoint_dir = '../result'
### change dataInput here !!!
def loadData():
    train_data, train_label, eval_data, eval_label = mrcReader.loadInputData()
#    test_data = np.random.permutation(eval_data)
    test_data = eval_data
    return test_data

def test():
    test_data = loadData()
    
    test_data_node = tf.placeholder(
                    tf.float32,
                    shape=(batch_size, image_size, image_size, 1)
    )
    logits = deepModel.inference(test_data_node, train=False)
    pred = deepModel.loss(logits, None)

    def test_in_batch(data, sess):
        size = test_size
        predictions = np.ndarray(shape=(size, num_class), dtype=np.float32)
        for begin in xrange(0, size, batch_size):
            end = begin + batch_size
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    pred,
                    feed_dict={test_data_node: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    pred,
                    feed_dict={test_data_node: data[-batch_size:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("no model find")
        label = test_in_batch(test_data, sess)
        print(np.argmax(label,1))

def main(argv=None):
    test()
if __name__ == '__main__':
    tf.app.run()
