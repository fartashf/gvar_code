import tensorflow as tf
import numpy as np
from kfac_helper import *


def deconv2D_fstats_channel_fac(var, fpropFactor, bpropFactor, stats, 
                                diag_init_coeff=0., 
                                homogeneous_coord=True):

    # D x D covariance matrix
    fpropFactor_size = fpropFactor.get_shape()[-1]

    addStatsSlot(stats[var]['fprop_concat_stats'], fpropFactor_size, var, 
                  'KFAC_STATS/'+fpropFactor.op.name, diag_init_coeff)



def deconv2D_bstats_channel_fac(var, fpropFactor, bpropFactor, stats,  
                              diag_init_coeff = 0.,
                              homogeneous_coord=True, 
                              channel_fac = False):
    Kh = var.get_shape()[0]
    Kw = var.get_shape()[1]
    C = bpropFactor.get_shape()[-1]
    
    Oh = fpropFactor.get_shape()[1]
    Ow = fpropFactor.get_shape()[2]
    if Oh==1 and Ow==1 and channel_fac:
      # factorization along the channels
      # assume independence bewteen input channels and spatial
      # 2K-1 x 2K-1 covariance matrix and C x C covariance matrix
      # factorization along the channels do not support homogeneous coordinate, assnBias is always None
      bpropFactor2_size = Kh*Kw
      addStatsSlot(stats[var]['bprop_concat_stats'], bpropFactor2_size, var, 
                  'KFAC_STATS/'+bpropFactor.op.name, diag_init_coeff)
      
      bpropFactor_size = C
    else:
      # 2K-1 x 2K-1 x C x C covariance matrix
      # assume BHWC
      bpropFactor_size = Kh*Kw*C

    addStatsSlot(stats[var]['bprop_concat_stats'], bpropFactor_size, var, 
                  'KFAC_STATS/'+bpropFactor.op.name, diag_init_coeff)


def compute_deconv2D_stats(Factor, fprop, bprop, 
                           fops, var, config):
  ## deconv2D
  convkernel_size = var.get_shape()
  if fprop:
      factor_dim = int(convkernel_size[-1])
  if bprop:
      factor_dim = int(np.prod(convkernel_size[:3]))
  diag_approx = factor_dim > DIAG_APPROX_THRESHOLD
  if config['approxT2']:
    cov = compute_deconv2D_stats_T2(Factor, 
                                    fprop=fprop, bprop=bprop, 
                                    fops = fops, var=var, diag_approx=diag_approx)
  else:
    ## full rank
    cov = compute_deconv2D_stats_default(Factor, 
                                 fprop=fprop, bprop=bprop, 
                                 fops = fops,  var=var, diag_approx=diag_approx)

  return cov




def compute_deconv2D_stats_default(factorTensor, fprop=False, bprop=False, fops=None,  
                                   var=None, stats_var=None, homogeneous_coord=True, diag_approx=False):
    numObservation = tf.to_float(tf.shape(factorTensor)[0])

    strides = fops.get_attr("strides")
    padding = fops.get_attr("padding")
    #convkernel_size = var.get_shape()[0:3]
    convkernel_size = var.get_shape()

    KH = int(convkernel_size[0])
    KW = int(convkernel_size[1])
    Cout = int(convkernel_size[2])
    Cin = int(convkernel_size[3])
    flatten_size = int(KH*KW*Cout)
    

    if fprop:
      Oh = int(factorTensor.get_shape()[1])
      Ow = int(factorTensor.get_shape()[2])
      T = Oh*Ow
      ## factor is B x Oh x Ow x Cin
      coeff = np.sqrt(1./Oh/Ow)# T * 1/T terms
      factorTensor = tf.reshape(factorTensor, [-1, Cin])*coeff  # T * 1/T terms, AA/T
    if bprop:
      ## poor mem usage implementation
      patches = tf.extract_image_patches(factorTensor, ksizes=[1, KH, KW, 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)
      Oh = int(patches.get_shape()[1])
      Ow = int(patches.get_shape()[2])
      T = Oh*Ow
      coeff = 1.# T * 1/T terms
      factorTensor = tf.reshape(patches, [-1, flatten_size])*coeff # size: (B x Oh x Ow) x (kH x kW x Cout), DD*T^2

    if diag_approx:
      cov = tf.reduce_sum(tf.square(factorTensor), [0])/(numObservation)
    else:
      cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)/numObservation

    if KFAC_DEBUG and diag_approx:
      if fprop:
          print 'diagonal act cov for %s'%(factorTensor.name)
      if bprop:
          print 'diagonal grad cov for %s'%(factorTensor.name)
    return cov


def compute_deconv2D_stats_T2(factorTensor, fprop=False, bprop=False, fops=None, 
                              var=None, stats_var=None, homogeneous_coord=True, diag_approx=False):
    numObservation = tf.to_float(tf.shape(factorTensor)[0])

    strides = fops.get_attr("strides")
    padding = fops.get_attr("padding")
    #convkernel_size = var.get_shape()[0:3]
    convkernel_size = var.get_shape()

    KH = int(convkernel_size[0])
    KW = int(convkernel_size[1])
    Cout = int(convkernel_size[2])
    Cin = int(convkernel_size[3])
    flatten_size = int(KH*KW*Cout)
    
    if KFAC_DEBUG:
      if fprop:
          print 'approxT2 act cov for %s'%(factorTensor.name)
      if bprop:
          print 'approxT2 grad cov for %s'%(factorTensor.name)

    if fprop:
      Oh = int(factorTensor.get_shape()[1])
      Ow = int(factorTensor.get_shape()[2])
      T = Oh*Ow
      factorTensor = tf.reduce_mean(factorTensor, [1, 2]) # T^2 terms * 1/T^2, AA/T^2
    if bprop:
      ## poor mem usage implementation
      patches = tf.extract_image_patches(factorTensor, ksizes=[1, KH, KW, 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)
      Oh = int(patches.get_shape()[1])
      Ow = int(patches.get_shape()[2])
      T = Oh*Ow
      factorTensor = tf.reduce_sum(patches, [1, 2]) # T^2 terms * 1/T^2, DD*T^2

    if diag_approx:
      cov = tf.reduce_sum(tf.square(factorTensor), [0])/numObservation
    else:
      cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)/numObservation

    if KFAC_DEBUG and diag_approx:
      if fprop:
          print 'diagonal act cov for %s'%(factorTensor.name)
      if bprop:
          print 'diagonal grad cov for %s'%(factorTensor.name)

    return cov




