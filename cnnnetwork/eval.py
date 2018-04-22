# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import inferency

import train
#2
#2
#2

EVAL_INTERVAL_SEC = 10

def evalate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [mnist.validation.num_examples,
                                        inferency.IMAGE_SIZE,
                                        inferency.IMAGE_SIZE,
                                        inferency.NUM_CHANNELS], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, inferency.OUTPUT_NODE], name='y-input')

        y = inferency.interfence(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    xs, ys = mnist.validation.next_batch(mnist.validation.num_examples)
                    reshaped_xs = np.reshape(xs, (mnist.validation.num_examples,
                                                  inferency.IMAGE_SIZE,
                                                  inferency.IMAGE_SIZE,
                                                  inferency.NUM_CHANNELS))


                    validata_feed = {x: reshaped_xs, y_: ys}
                    accuracy_score = sess.run(accuracy, feed_dict=validata_feed)
                    print("After %s training steps,validation accuracy =%g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SEC)

def main(argv=None):
    mnist = input_data.read_data_sets("/home/zhangtx/ml/mnist/ministdata", one_hot=True)
    evalate(mnist)

if __name__ == '__main__':
    tf.app.run()


