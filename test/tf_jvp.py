import tensorflow as tf
import numpy as np
import time


if __name__ == '__main__':
    n_samples = 128
    dim = 10000
    X = tf.placeholder("float", [n_samples, dim])
    Y = tf.placeholder("float", [n_samples, 1])
    W = tf.Variable(tf.random_normal([dim, 1]))
    m = tf.Variable(tf.ones([dim, 1]))
    pred = tf.matmul(X, W)
    cost_ex = tf.pow(pred-Y, 2)/(2*n_samples)

    # Slow Jv
    # cost = tf.reduce_sum(cost_ex)
    # grads = tf.gradients(cost, W)[0]
    # jacob = tf.gradients(grads, X)[0]
    # jvp = tf.matmul(jacob, m)

    # Slow Jv
    Xi = tf.split(X, n_samples)
    Wi = [tf.identity(W) for x in Xi]
    pi = tf.stack([tf.matmul(x, w) for x, w in zip(Xi, Wi)])
    pi = tf.squeeze(pi, 2)
    cost = tf.pow(pi-Y, 2)/(2*n_samples)
    grads = tf.gradients(cost, Wi)
    jacob = tf.squeeze(tf.stack(grads))
    jvp = tf.matmul(jacob, m)

    # Fast Jv
    v = tf.ones_like(cost_ex)
    g = tf.gradients(cost_ex, W, grad_ys=v)
    jvp2 = tf.gradients(g, v, grad_ys=m)[0]

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        x = np.random.randn(n_samples, dim)
        y = np.ones((n_samples, 1))
        tjc = 0
        tjvp = 0
        tjvp2 = 0
        for i in range(10):
            tstart = time.time()
            jc = sess.run(jacob, feed_dict={X: x, Y: y})
            tjc = tjc + time.time()-tstart
            tstart = time.time()
            alpha = sess.run(jvp, feed_dict={X: x, Y: y})
            tjvp = tjvp + time.time()-tstart
            tstart = time.time()
            alpha2 = sess.run(jvp2, feed_dict={X: x, Y: y})
            tjvp2 = tjvp2 + time.time()-tstart
        print('jc: %.2fs, jvp: %.2fs, jvp2: %.2fs' %
              (tjc/10, tjvp/10, tjvp2/10))

    print(jacob.shape)

    # alpha and alpha2 should match
    print(alpha.shape)
    print(alpha2.shape)

    # print(alpha)
    # print(alpha2)
    print((alpha-alpha2).abs().max())
    import ipdb; ipdb.set_trace()  # NOQA
