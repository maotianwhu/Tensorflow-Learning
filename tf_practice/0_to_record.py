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
raw_img_dir = '/Volumes/zhulf/nibs_new/pre/label/day0_system_mito/images_raw'
anno_dir = '/Volumes/zhulf/nibs_new/pre/label/day0_system_mito/annos_raw'
listfile = '/Volumes/zhulf/nibs_new/pre/label/day0_system_mito/trainval_raw.txt'

f = open(listfile, 'r')
lines = f.readlines()

filenames = [os.path.join(raw_img_dir, filename.strip() + '.png') for filename in lines ]
annonames = [os.path.join(anno_dir, filename.strip() + '.png') for filename in lines ]

print(filenames)
print(annonames)

num_per_shard = int(math.ceil(len(filenames) / _NUM_SHARDS))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(raw_img_data, anno_img_data, filename):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(raw_img_data),
        'image/anno': bytes_feature(anno_img_data),
        'image/filename': bytes_feature(filename),
    }))

sess = tf.Session()
for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join('/Volumes/zhulf/nibs_new/pre/label/day0_system_mito/', str(shard_id) + '_' + str(_NUM_SHARDS) + '.tfrecord')
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))

        for i in range(start_ndx, end_ndx):
            try:
                sys.stdout.write("\r>> Conver images %d/%d shard %d" %(i+1, len(filenames), shard_id))
                sys.stdout.flush()
                #raw_img_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                raw_img_data = Image.open(filenames[i])
                raw_img_data = raw_img_data.tobytes()
                # anno_data = tf.gfile.FastGFile(annonames[i],'r').read()
                anno_data = Image.open(annonames[i])
                anno_data = anno_data.tobytes()
                #print(anno_data)
                example = image_to_tfexample(raw_img_data, anno_data, os.path.basename(filenames[i].strip()))
                tfrecord_writer.write(example.SerializeToString())
            except IOError as e:
                print("Could not read: " + filenames[i])
                print("Error: " + e)
                print("Skip it\n")





