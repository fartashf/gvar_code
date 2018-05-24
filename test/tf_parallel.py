import tensorflow as tf
import numpy as np


def cost_function(pred, Y):
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    return cost


if __name__ == '__main__':
    n_samples = 128
    dim = 100
    X = tf.placeholder("float", [n_samples, dim])
    Y = tf.placeholder("float", [n_samples, 1])
    W = tf.Variable(tf.random_normal([dim, 1]))
    # pred = tf.matmul(X, W)

    examples = tf.split(X, n_samples)
    weight_copies = [tf.identity(W) for x in examples]
    output = tf.stack([tf.matmul(x, w)
                       for x, w in zip(examples, weight_copies)])
    cost = cost_function(output, Y)
    per_example_gradients = tf.gradients(cost, weight_copies)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        x = np.ones((n_samples, 100))
        y = np.ones((n_samples, 1))
        g = sess.run(per_example_gradients, feed_dict={X: x, Y: y})

    print(len(g))
    print(g[0].shape)
    import ipdb; ipdb.set_trace()  # NOQA
