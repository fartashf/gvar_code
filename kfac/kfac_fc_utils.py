import tensorflow as tf
import numpy as np
from kfac_helper import *


def compute_matmul_stats(Factor, fprop=False, bprop=False, config=None):
    ## MatMul
    factor_dim = int(Factor[-1].get_shape()[-1]) if isinstance(Factor, tuple) else int(Factor.get_shape()[-1])
    diag_approx = (factor_dim > DIAG_APPROX_THRESHOLD)
    cov = compute_fc_stats(Factor, 
                             fprop=fprop, bprop=bprop, 
                             homogeneous_coord = config['useHomogeneous'],
                             diag_approx = diag_approx)
    return cov

def compute_fc_stats(factorTensor, fprop=False, bprop=False, homogeneous_coord = False, diag_approx = False):
  if fprop==True and bprop==True:
    ## full diag approx
    if isinstance(factorTensor, tuple):
      A, B = factorTensor
      numObservation = tf.to_float(tf.shape(B)[0])
      diag_cov = tf.matmul(A**2, B**2, transpose_a=True) / numObservation
    else:
      B = factorTensor
      numObservation = tf.to_float(tf.shape(B)[0])
      diag_cov = tf.reduce_mean(B**2, 0)
    return diag_cov
  else: 
    ## pad activations
    if homogeneous_coord and fprop:
      factorTensor = tf.concat(axis=1, values=[factorTensor, tf.ones([tf.shape(factorTensor)[0],1])]) ## pad ones

    numObservation = tf.to_float(tf.shape(factorTensor)[0])
    if fprop:
      factorTensor /= tf.sqrt(numObservation) ## for numerical statiblity
      if diag_approx:
        print('use diagonal act stats for {}'.format(factorTensor.name))
        cov = tf.reduce_mean(tf.square(factorTensor), [0])
      else:
        cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)
    else:
      if diag_approx:
        print('use diagonal grad stats for {}'.format(factorTensor.name))
        cov = tf.reduce_mean(tf.square(factorTensor), [0])/numObservation
      else:
        cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)/numObservation
    return cov


