#coding=utf-8
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/', one_hot=True)

numClasses = 10
inputSize = 784
numHiddenUnits = 50
trainingInterations = 10000
batchSize = 100

X = tf.placeholder(tf.float32, shape = [None, inputSize])
y = tf.placeholder(tf.float32, shape = [None, numClasses])

W1 = tf.Variable(tf.random_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, shape = [numHiddenUnits]))

W2 = tf.Variable(tf.random_normal([numHiddenUnits, numClasses], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, shape = [numClasses]))

hiddenLayerOutput = tf.matmul(X, W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)

finalOutput = tf.matmul(hiddenLayerOutput, W2) + B2
finalOutput = tf.nn.relu(finalOutput)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits = finalOutput))
opt = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(loss)

correct_predciton = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predciton, "float"))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingInterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]

    _, trainingLoss = sess.run([opt, accuracy], feed_dict={X: batchInput, y: batchLabels})

    if i%1000 == 0:
        train_accuracy = accuracy.eval(session = sess, feed_dict = {X: batchInput, y: batchLabels})
        print("step %d, training accuracy %g, %g" %(i, train_accuracy, trainingLoss))


batch = mnist.test.next_batch(batchSize)
testAccuracy = sess.run(accuracy, feed_dict={X: batch[0], y:batch[1]})

print("test accuracy %g" % (testAccuracy))
