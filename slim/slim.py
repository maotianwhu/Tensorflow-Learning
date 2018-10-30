#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Model Variables

weights = slim.model_variable('weights',
                              shape=[10, 10, 3, 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')

model_variables = slim.get_model_variables()

# Regular variables
my_var = slim.variable('my_var',
                       shape=[20, 1],
                       initializer=tf.zeros_initializer())

regular_variables_and_model_variables = slim.get_variables()

# 变量分为两类： 模型变量和局部变量。局部变量是不作为模型参数保存的，而模型变量会在save的时候保存下来。
# 诸如global_step之类的就是局部变量
# slim中可以写明变量存放的设备，正则和初始化规则
# 获取变量的函数也需要注意一下， get_variables是返回所有的变量

# --------------------------------------------------------------------------------------------

# layer

# by tensorflow
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name="weights")
    conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name="biases")
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)

# by slim
input = ...
net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')

# ------------------------------------------------------

net = ...
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')

# 在slim中的repeat操作可以减少代码量

net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope="conv3")
net = slim.max_pool2d(net, [2, 2], scope='pool2')

# ------------------------------------------------------


# stack 是处理卷积核或者输出不一样的情况
x = slim.fully_connected(x, 32, scope='fc/fc_1')
x = slim.fully_connected(x, 64, scope='fc/fc_2')
x = slim.fully_connected(x, 128, scope='fc/fc3')

slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')


x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')

slim.stack(x, slim.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])])


# --------------------------------------------------------------------------------------------

# argscope
# 如果你对网络有大量相同的参数
slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
slim.conv2d(net, 128, [11, 11], padding="VALID",
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')

slim.conv2d(net, 256, [11, 11], padding='SAME',
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')

# 然后我们用arg_scope处理一下:

with slim.arg_scope([slim.conv2d], padding='SAME',
                    weights_initializer = tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer = slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 64, [11, 11], scope="conv1")
    net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
    net = slim.conv2d(net, 256, [11, 11], scope="conv3")

# arg_scope的作用范围内， 是定义了指定层的默认参数， 若想特别指定某些层的参数， 可以重新赋值(相当于重写)


# 如果除了卷积层还有其他层呢？ 那就要如下定义:

with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn = tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    with slim.arg_scope([slim.conv2d], strides=1, padding='SAME'):
        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
        net = slim.conv2d(net, 256, [5, 5],
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                          scope='conv2')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')



# -------------------------------------------------------------------------------------------------
def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer = slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        net = slim.fully_connected(net, 4096, scope="fc6")
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope="fc7")
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')

    return net



# --------------------------------------------------------------------------------------------------
# train the model

import tensorflow as tf
vgg = tf.contrib.slim.nets.vgg

images, labels = ...

predictions, _ = vgg.vgg_16(images)
loss = slim.losses.softmax_cross_entropy(predictions, labels)

# 关于loss, 要说一下定义自己的loss的方法，以及注意不要忘记加入到slim中，让slim看到你的loss

images, scene_labels, depth_labels, pose_labels = ...

scene_predictions, depth_prediction, pose_predictions = CreateMultiTaskModel(images)

classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.softmax_cross_entropy(depth_prediction, depth_labels)
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
slim.losses.add_loss(pose_loss)

# The following two ways to compute the total loss are equivalent:
regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
total_loss = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss

total_loss2 = slim.losses.get_total_loss()

#-----------------------------------------------------------------------------------------------------
# create some variables
v1 = slim.variable(name="v1", ...)
v2 = slim.variable(name="nested/v2")

# get list of variables to restore (which contains only 'v2')
variables_to_restore = slim.get_variables_by_name("v2")

restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    restorer.restore(sess, "/tmp/model.ckpt")
    print("Model restored")

# 除了这种部分变量加载的方法外，我们甚至还能加载到不同名字的变量中。
# 假设我们定义的网络变量是conv1/weights，而从VGG加载的变量名为vgg16/conv1/weights，
# 正常load肯定会报错（找不到变量名），但是可以这样：

def name_in_checkpoint(var):
    return 'vgg16/' + var.op.name


variables_to_restore = slim.get_model_variables()
variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore}
restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    # Restore variables from disk.
    restorer.restore(sess, "/tmp/model.ckpt")

#通过这种方式， 我们可以加载不同变量名的变量
















