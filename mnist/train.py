# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import inference
from tensorflow.examples.tutorials.mnist import input_data
import time

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/home/zhangtx/ml/mnist/model"
MODEL_NAME = "model.ckpt"

def train(mnist):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())


    with tf.name_scope("loss_function"):
        cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entroy_mean = tf.reduce_mean(cross_entroy)
        loss = cross_entroy_mean + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./logs/", tf.get_default_graph())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            time.sleep(1)
            if i % 100 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: xs, y_: ys},
                                               options=run_options,
                                               run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%03d'%i)
                print("After %d training steps,loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: xs, y_: ys})


    writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets("/home/zhangtx/ml/mnist/ministdata",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()












