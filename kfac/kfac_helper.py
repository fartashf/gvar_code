import tensorflow as tf
import numpy as np
KFAC_DEBUG = False
DIAG_APPROX_THRESHOLD = 2000

KFAC_OPS = ['MatMul', 'Conv2D', 'BiasAdd', 'Conv2DBackpropInput']
KFAC_GRAD_OPS = ['MatMul', 'Conv2DBackpropFilter', 'BiasAdd', ]
KFAC_GRAD_OPS_LOOKUP = {'MatMul':'MatMul', 'Conv2DBackpropFilter':'Conv2D', 'BiasAdd':'BiasAdd', }
homogenous_KFAC_OPS = ['MatMul', 'Conv2D', ]

def dampFactors(whiten_matrices, damping_coeff):
    flattened_dict = whiten_matrices.items()
    eigen_b, eigen_f = flattened_dict[0][1], flattened_dict[1][1]
    e_b, e_f = eigen_b['e'], eigen_f['e']
    sqrt_damping_coeff = tf.sqrt(damping_coeff)
    damping_correction = tf.sqrt(tf.reduce_mean(tf.abs(e_f)) / tf.reduce_mean(tf.abs(e_b)))
    e_b += sqrt_damping_coeff / damping_correction
    e_f += sqrt_damping_coeff * damping_correction
    eigen_b['e'], eigen_f['e'] = e_b, e_f
    return whiten_matrices

def updateRunningAverage(tensor, tensor_var, coeff, num_steps=None):
    bias_correction = 1.
    #if num_steps is not None:
    #  bias_correction *= 1. - tf.cond(tf.equal(num_steps, 0), lambda:tf.constant(0.), lambda: tf.pow(coeff, tf.to_float((num_steps))))
    #  bias_correction /= 1. - tf.pow(coeff, tf.to_float(num_steps+1))
      #bias_correction = tf.Print(bias_correction, [bias_correction,])
    new_tensor_var_val = (tensor_var * coeff + tensor * (1.-coeff)) * bias_correction
    assign_op = tf.assign(tensor_var, new_tensor_var_val, use_locking=True)
    return assign_op, new_tensor_var_val

def tensor_name(tensor):
    name = tensor.name.split(':')[0]
    return name 

def createSlotLike(covar, diag_init_coeff, createdSlots=None, name=None):
    if createdSlots is None:
      createdSlots = {}
    #########
    def createSlot(covar, name):
      num_dim = len(covar.get_shape())
      size = int(covar.get_shape()[0])
      if name is None:
        name = 'slot'
      assert num_dim<=2, "something is wrong with stats computation {}".format(covar.name)
      with tf.device('/cpu:0'):
        if num_dim == 1:
          initializer = tf.ones_like(covar)*diag_init_coeff
        elif num_dim == 2:
          initializer = tf.diag(tf.ones([size]))*diag_init_coeff
        stats_slot = tf.get_variable(name, initializer = initializer, trainable=False)
      return stats_slot
    ########
    if not createdSlots.has_key(covar):
      slot = createSlot(covar, name)
      createdSlots[covar] = slot
    else:
      slot = createdSlots[covar]
    return slot


def mergeFactors(factors_old, factors_new):
    for k,v in factors_new.items():
        if k not in factors_old:
            factors_old[k] = v
        else:
            for kk, vv in v.items():
              if 'slot' not in kk or 'inverse' not in kk:
                factors_old[k][kk] = vv
    return factors_old

