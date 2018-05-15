import tensorflow as tf
import numpy as np
from kfac_helper import *


def conv2D_fstats_channel_fac(var, fpropFactor, bpropFactor, stats,  
                              diag_init_coeff = 0.,
                              homogeneous_coord=True, 
                              channel_fac = False):
    Kh = var.get_shape()[0]
    Kw = var.get_shape()[1]
    C = fpropFactor.get_shape()[-1]
    
    Oh = bpropFactor.get_shape()[1]
    Ow = bpropFactor.get_shape()[2]
    if Oh==1 and Ow==1 and channel_fac:
      # factorization along the channels
      # assume independence bewteen input channels and spatial
      # 2K-1 x 2K-1 covariance matrix and C x C covariance matrix
      # factorization along the channels do not support homogeneous coordinate, assnBias is always None
      fpropFactor2_size = Kh*Kw
      addStatsSlot(stats[var]['fprop_concat_stats'], fpropFactor2_size, var, 
                  'KFAC_STATS/'+fpropFactor.op.name, diag_init_coeff)
      
      fpropFactor_size = C
    else:
      # 2K-1 x 2K-1 x C x C covariance matrix
      # assume BHWC
      fpropFactor_size = Kh*Kw*C

    # use homogeneous coordinate
    if homogeneous_coord and stats[var]['assnBias']:
      fpropFactor_size += 1
    
    addStatsSlot(stats[var]['fprop_concat_stats'], fpropFactor_size, var, 
                  'KFAC_STATS/'+fpropFactor.op.name, diag_init_coeff)


def removeHomogenousChannelFac(varlist, factors):
    for var in varlist:
       fpropFactor = factors[var]['fpropFactors_concat']
       bpropFactor = factors[var]['bpropFactors_concat']
       opType = factors[var]['opName']
       if opType == 'Conv2D':
           Kh = var.get_shape()[0]
           Kw = var.get_shape()[1]
           C = fpropFactor.get_shape()[-1]
           
           Oh = bpropFactor.get_shape()[1]
           Ow = bpropFactor.get_shape()[2]
           if Oh==1 and Ow==1:
             # factorization along the channels do not support homogeneous coordinate
             var_assnBias = factors[var]['assnBias']
             if var_assnBias:
               factors[var]['assnBias'] = None
               factors[var_assnBias]['assnWeights'] = None



def compute_conv2D_stats(Factor, fprop, bprop, 
                         fops, var, config):

  factor_shape = Factor[-1].get_shape() if isinstance(Factor, tuple) else Factor.get_shape()
  ## conv2D
  Oh = int(factor_shape[1])
  Ow = int(factor_shape[2])
  if Oh==1 and Ow==1 and channel_fac:
  ## rank1 approx
    SVD_factors = compute_svd_fstats(var_factors, Factor, fops)
    cov = compute_conv2D_1x1_svd_fstats(SVD_factors, 
                                        fprop=fprop, bprop=bprop, 
                                        fops = fops, 
                                        homogeneous_coord = config['useHomogeneous'])
  else:
    convkernel_size = var.get_shape()
    if fprop:
      factor_dim = int(np.prod(convkernel_size[:3]))
      #factor_dim = int(np.prod(convkernel_size[1:]))
    if bprop:
      factor_dim = int(convkernel_size[-1])

    diag_approx = factor_dim > DIAG_APPROX_THRESHOLD
    if config['approxT2']:
      ## rank1
      cov = compute_conv2D_stats_T2(Factor, 
                                    fprop=fprop, bprop=bprop, 
                                    fops = fops, var=var, 
                                    homogeneous_coord = config['useHomogeneous'],
                                    diag_approx=diag_approx)
    else:
      ## full rank
      cov = compute_conv2D_stats_default(Factor, 
                                 fprop=fprop, bprop=bprop, 
                                 fops = fops,  var=var,
                                 homogeneous_coord = config['useHomogeneous'],
                                 diag_approx=diag_approx)

  return cov




def compute_conv2D_stats_default(factorTensor, fprop=False, bprop=False, fops=None,  var=None, stats_var=None, homogeneous_coord=True, diag_approx=False):

  numObservation = tf.to_float(tf.shape(factorTensor[-1])[0] if isinstance(factorTensor, tuple) else tf.shape(factorTensor)[0])
  strides = fops.get_attr("strides")
  padding = fops.get_attr("padding")
  #convkernel_size = var.get_shape()[0:3]
  convkernel_size = var.get_shape()

  KH = int(convkernel_size[0])
  KW = int(convkernel_size[1])
  Cin = int(convkernel_size[2])
  Cout = int(convkernel_size[3])
  flatten_size = int(KH*KW*Cin)

  if fprop==True and bprop==True:
  ## full diag approx
    A, B = factorTensor 
    patches = tf.extract_image_patches(A, ksizes=[1, KH, KW, 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)
    flatten_A = tf.reshape(patches, [-1, flatten_size]) # size: (B x Oh x Ow) x (Kh x Kw x C)
    flatten_B = tf.reshape(B, [-1, Cout]) # size: (B x Oh x Ow) x (Kh x Kw x C)
    diag_cov = tf.matmul(flatten_A**2, flatten_B**2, transpose_a=True) / numObservation
    diag_cov = tf.reshape(diag_cov, convkernel_size)
    return diag_cov
  else:
  ## kfac approx
    if fprop:
      ## poor mem usage implementation
      patches = tf.extract_image_patches(factorTensor, ksizes=[1, KH, KW, 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)
      Oh = int(patches.get_shape()[1])
      Ow = int(patches.get_shape()[2])
      T = Oh*Ow
      coeff = tf.sqrt(1./numObservation/T)# T * 1/T terms
      factorTensor = tf.reshape(patches, [-1, flatten_size])*coeff # size: (B x Oh x Ow) x (Kh x Kw x C)

      ## pad activations
      if homogeneous_coord:
        factorTensor = tf.concat(axis=1, values=[factorTensor, tf.ones([tf.shape(factorTensor)[0],1])*coeff])

      if diag_approx:
        cov = tf.reduce_sum(tf.square(factorTensor), [0])
      else:
        cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)

    if bprop:
      Oh = int(factorTensor.get_shape()[1])
      Ow = int(factorTensor.get_shape()[2])
      T = Oh*Ow
      coeff = np.sqrt(T)# T * 1/T terms
      #coeff = 1.# T * 1/T terms
      factorTensor = tf.reshape(factorTensor, [-1, Cout])*coeff  # T * 1/T terms
      if diag_approx:
        cov = tf.reduce_sum(tf.square(factorTensor), [0])/numObservation/T
      else:
        cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)/numObservation/T

    if KFAC_DEBUG and diag_approx:
      if fprop:
          print 'diagonal act cov for %s'%(factorTensor.name)
      if bprop:
          print 'diagonal grad cov for %s'%(factorTensor.name)

    return cov


def compute_conv2D_stats_T2(factorTensor, fprop=False, bprop=False, fops=None, var=None, stats_var=None, homogeneous_coord=True, diag_approx=False):
    numObservation = tf.to_float(tf.shape(factorTensor)[0])

    strides = fops.get_attr("strides")
    padding = fops.get_attr("padding")
    #convkernel_size = var.get_shape()[0:3]
    convkernel_size = var.get_shape()

    KH = int(convkernel_size[0])
    KW = int(convkernel_size[1])
    Cin = int(convkernel_size[2])
    Cout = int(convkernel_size[3])
    flatten_size = int(KH*KW*Cin)
    
    if KFAC_DEBUG:
      if fprop:
          print 'approxT2 act cov for %s'%(factorTensor.name)
      if bprop:
          print 'approxT2 grad cov for %s'%(factorTensor.name)
    if fprop:
      ## poor mem usage implementation
      patches = tf.extract_image_patches(factorTensor, ksizes=[1, KH, KW, 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)
      Oh = int(patches.get_shape()[1])
      Ow = int(patches.get_shape()[2])
      T = Oh*Ow
      factorTensor = tf.reduce_mean(patches, [1, 2]) # T^2 terms * 1/T^2

      ## pad activations
      if homogeneous_coord:
        factorTensor = tf.concat(axis=1, values=[factorTensor, tf.ones([tf.shape(factorTensor)[0],1])])

    if bprop:
      Oh = int(factorTensor.get_shape()[1])
      Ow = int(factorTensor.get_shape()[2])
      T = Oh*Ow
      factorTensor = tf.reduce_sum(factorTensor, [1, 2]) # T^2 terms * 1/T^2

    if KFAC_DEBUG and diag_approx:
      if fprop:
          print 'diagonal act cov for %s'%(factorTensor.name)
      if bprop:
          print 'diagonal grad cov for %s'%(factorTensor.name)
    if diag_approx:
      cov = tf.reduce_sum(tf.square(factorTensor), [0])/numObservation
    else:
      cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)/numObservation

    return cov



def compute_conv2D_block_stats(factorTensor, fprop=False, bprop=False, fops=None, var=None):
  Cout = int(factorTensor[-1].get_shape()[-1] if isinstance(factorTensor, tuple) else factorTensor.get_shape()[-1])  #num channels
  numObservation = tf.to_float(tf.shape(factorTensor[-1])[0] if isinstance(factorTensor, tuple) else tf.shape(factorTensor)[0])
  if KFAC_DEBUG:
        print 'block diag approx cov for %s'%(factorTensor.name)
  convkernel_size = var.get_shape()
  factor_dim = int(convkernel_size[-1])

  if fprop==True and bprop==True:
  ## full diag approx
    B = factorTensor 
    flatten_B = tf.reshape(B, [-1, Cout]) # size: (B x Oh x Ow) x (Kh x Kw x C)
    diag_cov = tf.reduce_sum(flatten_B**2, [0]) / numObservation
    diag_cov = tf.reshape(diag_cov, convkernel_size)
    return diag_cov

  else:
    assert fprop == False, "can not compute forward stats for {}".format(factorTensor.name)
    diag_approx = factor_dim > DIAG_APPROX_THRESHOLD
    if bprop:
      factorTensor = tf.reduce_sum(factorTensor, [1,2]) # per image gradient
    if diag_approx:
      cov_b = tf.reduce_sum(tf.square(factorTensor), [0])/numObservation
    else:
      cov_b = tf.matmul(factorTensor, factorTensor, transpose_a=True)/numObservation
    return cov_b


def compute_conv2D_1x1_svd_fstats(SVD_factors, fprop=False, bprop=False, fops=None, stats_var=None, homogeneous_coord=True):
    stats_var_dim = int(stats_var.get_shape()[0])
    factorTensor = SVD_factors[stats_var_dim]

    numObservation = tf.to_float(tf.shape(factorTensor)[0])
    factorTensor_size = int(fpropFactor.get_shape()[-1])

    ### use homogeneous coordinates
    if homogeneous_coord and fprop:
      factorTensor = tf.concat(axis=1, values=[factorTensor, tf.ones([tf.shape(factorTensor)[0],1])])
    cov = tf.matmul(factorTensor, factorTensor, transpose_a=True)/numObservation
    return cov


def compute_svd_fstats(factor, fpropFactor, fops):
    if not factor.has_key('SVD_factors'):
      factor['SVD_factors'] = SVD_factors = {}
      strides = fops.get_attr("strides")
      padding = fops.get_attr("padding")
      convkernel_size = var.get_shape()[0:3]

      KH = int(convkernel_size[0])
      KW = int(convkernel_size[1])
      Cin = int(convkernel_size[2])
      Cout = int(convkernel_size[3])
      flatten_size = int(KH*KW*C)
      ## factorization along the channels
      ## assume independence among input channels
      ## only apply this to a single large 1x1 conv layer
      ## factor = B x 1 x 1 x (KH xKW x C)
      ## patches = B x Oh x Ow x (KH xKW x C)
      #factorTensor = tf.extract_image_patches(factorTensor, ksizes=[1, convkernel_size[0], convkernel_size[1], 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)
      Oh = int(factorTensor.get_shape()[1])
      Ow = int(factorTensor.get_shape()[2])
      S, U, V = tf.svd(tf.reshape(fpropFactor, [-1, KH*KW, C]))
      ## get rank-1 approx slides
      sqrtS1 = tf.expand_dims(tf.sqrt(S[:,0,0]), 1)
      patches_k = U[:, :, 0]*sqrtS1  # B x KH*KW
      full_factor_shape = fpropFactor.get_shape()
      patches_k.set_shape([full_factor_shape[0], KH*KW])
      patches_c = V[:, :, 0]*sqrtS1  # B x C
      patches_c.set_shape([full_factor_shape[0], C])
      SVD_factors[C] = patches_c
      SVD_factors[KH*KW] = patches_k
    
    return factor['SVD_factors']



