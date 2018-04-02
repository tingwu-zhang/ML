# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
vector_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 3 + 0.3 + np.random.normal(-0.5, 0.5)
    vector_set.append([x1, y1])

x_data = [v[0] for v in vector_set]
x_data = np.reshape(x_data, (1000, 1))
y_data = [v[1] for v in vector_set]
y_data = np.reshape(y_data, (1000, 1))
plt.scatter(x_data, y_data, c='r')
plt.show()

W = tf.Variable(tf.random_uniform([1, 1], -1, 1, dtype=tf.float32), name="weight")
b = tf.Variable(tf.zeros([1, 1]), name="b", dtype=tf.float32)

x_ = tf.placeholder(tf.float32, [None, 1], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 1], name="y-input")

y = tf.add(tf.matmul(x_, W), b)
loss = tf.reduce_mean(tf.square(y-y_), name="loss")

train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for times in range(100000):
        _ = sess.run([train], feed_dict={x_:x_data, y_:y_data})
        loss_ = sess.run([loss], feed_dict={x_:x_data, y_:y_data})
        print sess.run(W), sess.run(b)


        plt.scatter()
