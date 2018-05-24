import tensorflow as tf


# Jv: jacobian vector product, same dimension as per_data_point_loss
# vec: could be g_bar


def fwd_gradients(ys, xs, d_xs):
    """
    Forward-mode pushforward analogous to the pullback defined by tf.gradients.
    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs
    is the vector being pushed forward.
    """
    v = tf.ones_like(ys)  # dummy variable
    g = tf.gradients(ys, xs, grad_ys=v)
    return tf.gradients(g, v, grad_ys=d_xs)


# a vector of length = num data points
per_data_point_loss = self.loss_per_example
Jv = fwd_gradients(per_data_point_loss
                   / tf.to_float(tf.shape(per_data_point_loss)[0]),
                   var, vec)  # jacobian vector product
