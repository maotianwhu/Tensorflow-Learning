#coding=utf-8
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/', one_hot=True)


print ("类型是 %s" % type(mnist))

print ("训练数据集有 %d " %(mnist.train.num_examples))
print ("测试数据有 %d" %(mnist.test.num_examples))

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print("数据类型 is %s" %(type(trainimg)))
print("标签类型 is %s" %(type(trainlabel)))

print("训练集的shape %s" %(trainimg.shape, ))
print("测试集的shape %s" %(testimg.shape, ))

print("训练集标签的shape %s" %(trainlabel.shape,))
print("测试集标签的shape %s" %(testlabel.shape,))

img = np.reshape(trainimg[8, :], (28, 28))
plt.matshow(img, cmap=plt.get_cmap('gray'))

plt.show()

print(np.argmax(trainlabel[8, :]))

batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)

print(type(batch_xs))
print(type(batch_ys))
print(batch_xs.shape)
print(batch_ys.shape)
