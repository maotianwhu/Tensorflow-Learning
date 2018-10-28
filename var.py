import tensorflow as tf
import numpy as np

w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])

y = tf.matmul(w, x)

init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)
    print y.eval()


norm = tf.random_normal([2, 3], mean=-1, stddev=4)

c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

sess = tf.Session()
print sess.run(norm)
print sess.run(shuff)

state = tf.Variable(0)
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(state)
    for _ in range(3):
        sess.run(update)
        print sess.run(state)


a = np.zeros((3, 3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print sess.run(ta)


a = tf.constant(5.0)
b = tf.constant(10.0)

x = tf.add(a, b)
y = tf.div(a, b)

with tf.Session() as sess:
    print sess.run(x)
    print sess.run(y)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print sess.run([output], feed_dict={input1: [7.], input2:[2.]})


