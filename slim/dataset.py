#coding=utf-8

# 通过slim读取生成的tfrecord, 读取和解码操作由tf.TFRecordRead完成

import tensorflow as tf
slim = tf.contrib.slim

file_pattern = './pascal_train_*.tfrecord'

# 适配器1: 将examples 反序列化成存储之前的格式。 由 tf 完成

keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/height': tf.FixedLenFeature([1], tf.int64),
    'image/width': tf.FixedLenFeature([1], tf.int64),
    'image/channel': tf.FixedLenFeature([1], tf.int64),
    'image/channels': tf.FixedLenFeature([3], tf.int64),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.VarLenFeature(dtype=tf.float64),
    'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
}

# 适配器2: 将反序列化的数据组装成更高级的格式。 由slim完成
items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'shape': slim.tfexample_decoder.Tensor('image/shape'),
    'object/bbox': slim.tfexample_decoder.BoundingBox(
        ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
    'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
    'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
}

# 解码器
decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

# dataset对象定义了数据集的文件位置， 解码方式等元信息

dataset = slim.dataset.Dataset(
    data_sources = file_pattern,
    reader  =tf.TFRecordReader,
    num_samples = 3, # 手动生成了三个文件，每个文件里只包含一个example
    decoder = decoder,
    items_to_descriptions = {},
    num_classes = 21)

# provider 对象根据dataset信息读取数据
provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers = 3,
    shuffle = False
)

[image, shape, labels, gbboxes] = provider.get(['image',
                                                'shape',
                                                'object/label',
                                                'object/bbox'])

