# coding=utf-8

# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import math
import sys

from PIL import Image


_NUM_SHARDS = 4
BATCH_SIZE = 8

data_dir = '/Volumes/zhulf/nibs_new/pre/label/day0_system_mito/'

filelist = [os.path.join(data_dir, str(shard_id) + '_' + str(_NUM_SHARDS) + '.tfrecord') for shard_id in range(_NUM_SHARDS)]

print(filelist)

def read_and_decode(filelist):
    filename_queue = tf.train.string_input_producer(filelist)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/anno': tf.FixedLenFeature([], tf.string),
                                           'image/filename': tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [1887, 2048, 3])

    anno = tf.decode_raw(features['image/anno'], tf.uint8)
    anno = tf.reshape(anno, [1887, 2048])

    filename = features['image/filename']

    return image, anno, filename

image, anno, filename = read_and_decode(filelist)

image_batch, anno_batch, filename = tf.train.shuffle_batch([image, anno, filename], batch_size=BATCH_SIZE, capacity=128, min_after_dequeue=64, num_threads=4)
# image_batch, anno_batch, filename = tf.train.batch([image, anno, filename], batch_size=BATCH_SIZE, capacity=128)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(5):
        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        print(b_image.shape)
        print(b_anno.shape)
        print(b_filename)

    coord.request_stop()
    # 其他所有线程关闭后，这一函数才能返回
    coord.join(threads)