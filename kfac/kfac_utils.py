import tensorflow as tf
import numpy as np
from kfac_helper import *

from kfac_fc_utils import *
from kfac_conv_utils import *
from kfac_deconv_utils import *
from kfac_rnn_utils import *



def compute_default_xcovar(stats_var, var, factors, stats, fprop=False, bprop=False,
                           homogeneous_coord=True, channel_fac=False, 
                           approxT2=False,
                           temporal_dependency=False, stats_decay_rnn_dict=None):
    fops = factors[var]['op']
    fpropFactor = factors[var]['fpropFactors_concat']
    bpropFactor = factors[var]['bpropFactors_concat']

    ## assume sampled loss is averaged. TO-DO:figure out better way to handle this 
    if fprop:
        Factor = fpropFactor
    if bprop:
        ## assume sampled loss is averaged. TO-DO:figure out better way to handle this 
        factor_per_timestep = factors[var]['bpropFactors'][0]
        B = (tf.shape(factor_per_timestep)[0]) #batch size
        #B = (tf.shape(bpropFactor)[0]) #batch size
        bpropFactor *= tf.to_float(B)
        ##
        Factor = bpropFactor
    ##
    
    #########
    kfacType = factors[var]['KFACType']
    if kfacType == 'RNN' and temporal_dependency:
    ## correct for temporal dependency for the covariance matrix
      if factors[var]['opName'] == 'MatMul':
        if fprop:
          print('compute cross act stats temporal dependency for %s'%var.name)
        if bprop:
          print('compute cross grad stats temporal dependency for %s'%var.name)
        cov = compute_rnn_xstats(Factor, factors[var], stats[var], 
                                      fprop=fprop, bprop=bprop, 
                                      stats_var=stats_var, 
                                      homogeneous_coord = homogeneous_coord)
    #########

    #########
    ## dealing with weight sharing with rescaling
    fstats_list = factors[var]['fpropFactors']
    bstats_list = factors[var]['bpropFactors']
    if len(bstats_list) > 1:
      if len(fstats_list) == len(bstats_list):
        if bprop:
            cov *= np.single(len(bstats_list))
        #cov *= np.sqrt(len(bstats_list))
        #cov *= 1.
      elif len(fstats_list) == 0:
        cov *= np.single(len(bstats_list))
      else:
        print('error!!! something weird happened for weight sharing detection!!!')
        exit(0)
    #########
 
    return cov


def compute_stats_default(var, var_factors, fprop=False, bprop=False,
                          config=None, stats_decay_rnn_dict=None):
    fops = var_factors['op']
    opType = var_factors['opName']
    fpropFactor = var_factors['fpropFactors_concat']
    bpropFactor = var_factors['bpropFactors_concat']
    if fprop and bprop:
      ## Full diag approximation
      factor_per_timestep = var_factors['bpropFactors'][0]
      B = (tf.shape(factor_per_timestep)[0]) #batch size
      bpropFactor *= tf.to_float(B)
      if var_factors['fpropFactors_concat'] is not None:
        Factor = (fpropFactor, bpropFactor)
      else:
        Factor = bpropFactor

    else:
      if fprop:
          Factor = fpropFactor
      if bprop:
        ## assume sampled loss is averaged. 
        ## TO-DO:figure out better way to handle this 
        factor_per_timestep = var_factors['bpropFactors'][0]
        B = (tf.shape(factor_per_timestep)[0]) #batch size
        #B = (tf.shape(bpropFactor)[0]) #batch size
        bpropFactor *= tf.to_float(B)
        ##
        Factor = bpropFactor

    #if KFAC_DEBUG:
    if False:
        print('Factor for %s'%(var.name))
        print(fprop, bprop)
        print(Factor.get_shape())
        print(opType)
    #########
    numDims = len(Factor[-1].get_shape()) if isinstance(Factor, tuple) else len(Factor.get_shape())
    if numDims == 4:
      if opType == 'Conv2D':
        ## conv2D
        cov = compute_conv2D_stats(Factor, fprop=fprop, bprop=bprop, 
                                   fops=fops, var=var, config=config)
      elif opType == 'Conv2DBackpropInput':
        ## deconv2D
        cov = compute_deconv2D_stats(Factor, fprop=fprop, bprop=bprop, 
                                     fops = fops,  var=var, config=config)
      else:
        ## biasAdd and other ops, only compute bprop stats
        print('computing stats for {} with op {}'.format(var.name, var_factors['opName']))
        cov = compute_conv2D_block_stats(Factor, 
                                   fprop=fprop, bprop=bprop, 
                                   fops=fops, var=var)
    else:
      ## MatMul
      cov = compute_matmul_stats(Factor, 
                                 fprop=fprop, bprop=bprop, 
                                 config=config)
    #########
    

    #########
    ## dealing with weight sharing with rescaling
    ## Assume statistical independence among N gradients, 
    ## scale up F by N
    fstats_list = var_factors['fpropFactors']
    bstats_list = var_factors['bpropFactors']
    if len(bstats_list) > 1:
      if len(fstats_list) == len(bstats_list):
        if bprop:
            print('rescale {} matrix independent approx by a factor of {}'.format(var.name, len(bstats_list)))
            cov *= np.single(len(bstats_list))
        #cov *= np.sqrt(np.single(len(bstats_list)))
        #cov *= 1.
        #print('multiply a large number to cov') 
        #cov *= 10000.
      elif len(fstats_list) == 0:
        print('rescale {} bias independent approx by a factor of {}'.format(var.name, len(bstats_list)))
        cov *= np.single(len(bstats_list))
      else:
        print('error!!! something weird happened for weight sharing detection!!!')
        exit(0)
    #########
    
    return cov


def gmatmul(a, b, transpose_a = False, transpose_b = False, reduce_dim = None):
      if reduce_dim == None:
      #### general batch matmul
          if len(a.get_shape()) == 3 and len(b.get_shape()) == 3:
            return tf.matmul(a,b,adjoint_a = transpose_a, adjoint_b = transpose_b)
          elif len(a.get_shape()) == 3 and len(b.get_shape()) == 2:
            if transpose_b:
              N = b.get_shape()[0].value 
            else:
              N = b.get_shape()[1].value 
            B = a.get_shape()[0].value
            if transpose_a:
                K = a.get_shape()[1].value
                a = tf.reshape(tf.transpose(a, [0,2,1]), [-1, K]) 
            else:
                K = a.get_shape()[-1].value
                a = tf.reshape(a, [-1, K])
            result =  tf.matmul(a,b,transpose_b = transpose_b)
            result = tf.reshape(result, [B,-1,N])
            return result
          elif len(a.get_shape()) == 2 and len(b.get_shape()) == 3:
            if transpose_a:
              M = a.get_shape()[1].value 
            else:
              M = a.get_shape()[0].value 
            B = b.get_shape()[0].value
            if transpose_b:
                K = b.get_shape()[-1].value
                b = tf.transpose(tf.reshape(b, [-1, K]), [1,0])
            else:
                K = b.get_shape()[1].value
                b = tf.transpose(tf.reshape(tf.transpose(b, [0,2,1]), [-1, K]), [1,0])
            result =  tf.matmul(a,b,transpose_a = transpose_a)
            result = tf.transpose(tf.reshape(result, [M,B,-1]), [1,0,2])
            return result
          else:
            return tf.matmul(a,b,transpose_a = transpose_a, transpose_b = transpose_b)
      else:
      #### weird batch matmul
          if len(a.get_shape()) == 2 and len(b.get_shape()) > 2:
          ## reshape reduce_dim to the left most dim in b
              b_shape = b.get_shape()
              if reduce_dim != 0:
                b_dims = range(len(b_shape))
                b_dims.remove(reduce_dim)
                b_dims.insert(0, reduce_dim)
                b = tf.transpose(b, b_dims)
              b_t_shape = b.get_shape()
              b = tf.reshape(b, [int(b_shape[reduce_dim]), -1])
              result = tf.matmul(a,b,transpose_a = transpose_a, transpose_b = transpose_b)
              result = tf.reshape(result, b_t_shape)
              if reduce_dim != 0:
                b_dims = range(len(b_shape))
                b_dims.remove(0)
                b_dims.insert(reduce_dim, 0)
                result = tf.transpose(result, b_dims)
              return result

          elif len(a.get_shape()) > 2 and len(b.get_shape()) == 2:
          ## reshape reduce_dim to the right most dim in a
              a_shape = a.get_shape()
              outter_dim = len(a_shape) - 1
              reduce_dim = len(a_shape) - reduce_dim - 1
              if reduce_dim != outter_dim:
                a_dims = range(len(a_shape))
                a_dims.remove(reduce_dim)
                a_dims.insert(outter_dim, reduce_dim)
                a = tf.transpose(a, a_dims)
              a_t_shape = a.get_shape()
              a = tf.reshape(a, [-1, int(a_shape[reduce_dim])])
              result = tf.matmul(a,b,transpose_a = transpose_a, transpose_b = transpose_b)
              result = tf.reshape(result, a_t_shape)
              if reduce_dim != outter_dim:
                a_dims = range(len(a_shape))
                a_dims.remove(outter_dim)
                a_dims.insert(reduce_dim, outter_dim)
                result = tf.transpose(result, a_dims)
              return result

          elif len(a.get_shape()) == 2 and len(b.get_shape()) == 2:
              return tf.matmul(a,b,transpose_a = transpose_a, transpose_b = transpose_b)
            
          assert False, 'something went wrong'  



def clipoutNeg(vec, threshold=1e-6):
              mask = tf.cast(vec > threshold, tf.float32)
              return mask * vec

def detectMinVal(input_mat, var, threshold=1e-6, name='', debug=False):
              eigen_min = tf.reduce_min(input_mat)
              eigen_max = tf.reduce_max(input_mat)
              eigen_ratio = eigen_max/eigen_min
              input_mat_clipped = clipoutNeg(input_mat, threshold)

              if debug:
                input_mat_clipped = tf.cond(tf.logical_or(tf.greater(eigen_ratio, 0.), tf.less(eigen_ratio, -500)), lambda: input_mat_clipped, lambda: tf.Print(input_mat_clipped, [tf.convert_to_tensor('screwed ratio '+name+' eigen values!!!'), tf.convert_to_tensor(var.name), eigen_min, eigen_max, eigen_ratio]))
            
              return input_mat_clipped


def factorReshape(e, grad, facIndx=0, ftype='act'):
    grad_shape = grad.get_shape()
    if ftype == 'act':
        try:
          assert e.get_shape()[0] == grad_shape[facIndx]
        except:
          print(grad.name)
          print(e)
          print(grad)
          print(facIndx)
          exit()
        expanded_shape = [1,] * len(grad_shape)
        expanded_shape[facIndx] = -1
        e = tf.reshape(e, expanded_shape)
    if ftype == 'grad':
        assert e.get_shape()[0] == grad_shape[len(grad_shape) - facIndx - 1]
        expanded_shape = [1,] * len(grad_shape)
        expanded_shape[len(grad_shape) - facIndx - 1] = -1
        e = tf.reshape(e, expanded_shape)       

    return e
        

def getSampledFisher(factorTensors):
  for param in factorTensors:
    #if factorTensors[param]['bpropFactors_concat'] is not None and factorTensors[param]['opName'] != 'Conv2D':
    if factorTensors[param]['bpropFactors_concat'] is not None and len(factorTensors[param]['bpropFactors_concat'].get_shape())==2:
      if factorTensors[param]['fpropFactors_concat'] is not None:
        a = factorTensors[param]['fpropFactors_concat']
        b = factorTensors[param]['bpropFactors_concat']
        weight_size = int(a.get_shape()[1]) * int(b.get_shape()[1])
        ## grad = b^T a
        t = tf.reshape(tf.expand_dims(a, 2) * tf.expand_dims(b, 1), (-1, weight_size))
      else:
        t = factorTensors[param]['bpropFactors_concat']

      factorTensors[param]['fisherBlock_tensor']=t
      num_shared_copy = len(factorTensors[param]['bpropFactors'])
      if num_shared_copy > 1:
        print('summing gradient among the shared copies')
        weight_size = int(t.get_shape()[1])
        t = tf.reduce_sum(tf.reshape(t, (-1, num_shared_copy, weight_size)), 1)
      batch_size = tf.to_float(tf.shape(t)[0])
      t *= batch_size # the loss is averaged
      F = tf.matmul(t, t, transpose_a=True)/batch_size
      F_diag = tf.reduce_mean(tf.square(t), 0)
      factorTensors[param]['fisherBlock']=F
      factorTensors[param]['fisherBlock_diag']=F_diag
 
  return factorTensors


def reshapeGradients(grad, var, factor, config):
  channel_fac = config['channel_fac']
  grad_shape = grad.get_shape()
  var_shape = var.get_shape()
  GRAD_RESHAPE = (grad_shape == var_shape)

  opName = factor['opName']
  multiple_factors = True if factor.has_key('fpropFactors_concat_stats_slots') and \
                             isinstance(factor['fpropFactors_concat_stats_slots'], list) else False
  
  if GRAD_RESHAPE:
    #######################
    if len(grad.get_shape())>2:
      if opName == 'Conv2DBackpropInput':
      ## reshape deconv kernel parameters
        KW = int(grad.get_shape()[0])
        KH = int(grad.get_shape()[1])
        Cout = int(grad.get_shape()[2])
        Cin = int(grad.get_shape()[3])

        ## reshape deconv kernel parameters into 2D grad
        ## deconv kernel kW x kH x Cout x Cin
        grad = tf.transpose(tf.reshape(grad, [-1, Cin]))

      else:
      ## reshape conv kernel parameters
        KW = int(grad.get_shape()[0])
        KH = int(grad.get_shape()[1])
        C = int(grad.get_shape()[2])
        D = int(grad.get_shape()[3])

        if multiple_factors and channel_fac:
          ## reshape conv kernel parameters into tensor
          grad = tf.reshape(grad, [KW*KH, C, D])
        else:
          ## reshape conv kernel parameters into 2D grad
          grad = tf.reshape(grad, [-1, D])

    #######################
    elif len(grad.get_shape()) == 1:
    ## reshape bias or 1D parameters
      D = int(grad.get_shape()[0])
      
      grad = tf.expand_dims(grad,0)
    else:
    ## 2D parameters
      C = int(grad.get_shape()[0])
      D = int(grad.get_shape()[1])
  else:
    ## reshape back
    if opName == 'Conv2DBackpropInput':
        grad = tf.transpose(grad)

    grad = tf.reshape(grad, var_shape)
  return grad


def transform_grad(grad, Q_f, Q_b, transpose_f=True, transpose_b=False, reverse=False):
  ## transform gradient using both the left and the right
  ## eigenvectors
        for idx, (Q, transpose_a, transpose_b) in enumerate([(Q_f, transpose_f ^ reverse, False), (Q_b, False, transpose_b ^ reverse)]):
          if Q is not None:
            if len(Q.get_shape())==1:
                ## diagonal Kronecker sum
                Q = tf.expand_dims(Q, 1 - idx)
                grad *= Q
            else:
                if idx == 0:
                    a,b = Q, grad
                else:
                    a,b = grad, Q
                grad = gmatmul(a, b, transpose_a=transpose_a, 
                                        transpose_b=transpose_b, reduce_dim=idx)
       
  ##
        return grad



def transform_grad(grad, Q_f, Q_b, transpose_f=True, transpose_b=False, reverse=False):
  ## transform gradient using both the left and the right
  ## eigenvectors
        for idx, (Q, transpose_a, transpose_b) in enumerate([(Q_f, transpose_f ^ reverse, False), (Q_b, False, transpose_b ^ reverse)]):
          if Q is not None:
            if len(Q.get_shape())==1:
                ## diagonal Kronecker sum
                Q = tf.expand_dims(Q, 1 - idx)
                grad *= Q
            else:
                if idx == 0:
                    a,b = Q, grad
                else:
                    a,b = grad, Q
                grad = gmatmul(a, b, transpose_a=transpose_a, 
                                        transpose_b=transpose_b, reduce_dim=idx)
       
  ##
        return grad


def transform_to_eigenspace(grad, var, factor, reverse=False):
  ## transform gradient to eigen space and reshape the eigenvalues for broadcasting
  eigVals = []

  stats_list = [('fpropFactors_concat_stats_inverse', not reverse, False, 1),
                 ('bpropFactors_concat_stats_inverse', False, reverse, 0),]
  for key_name, transpose_a, transpose_b, grad_indx in stats_list:
    if factor.has_key(key_name):
        for idx, stats in enumerate([factor[key_name],]): ### quick dirty fix...
          if factor[key_name].has_key('Q_slot'):
            Q = factor[key_name]['Q_slot']
            if len(Q.get_shape())==1:
                ## diagonal Kronecker sum
                Q = tf.expand_dims(Q, grad_indx)
                grad *= Q
            else:
                if grad_indx == 1:
                    a,b = Q, grad
                else:
                    a,b = grad, Q
                grad = gmatmul(a, b, transpose_a=transpose_a, 
                                        transpose_b=transpose_b, reduce_dim=idx)
       
          if not reverse:
            if 'fprop' in key_name:
                name = 'act'
            if 'bprop' in key_name:
                name = 'grad'
            ## clip the numerical errors
            e = detectMinVal(factor[key_name]['e_slot'], var, name=name, debug=KFAC_DEBUG)
            ##
            e = factorReshape(e, grad, facIndx = idx, ftype=name)
            eigVals.append(e)
  ##
  return grad, eigVals



def compute_normalization_coeff(eigVals, var, epsilon, weightDecayCoeff, factored_damping, factor, use_KroneckerSum_dict):
  coeffs = 1.
  for e in eigVals:
      coeffs *= e
  #if factored_damping:
  #  if KFAC_DEBUG:
  #      print 'use factored damping for %s'%(var.name)
  #  coeffs = 1.
  #  num_factors = len(eigVals)
  #  ## compute the ratio of two trace norm of the left and 
  #  ## right KFac matrices, and their generalization
  #  if len(eigVals) == 1:
  #    damping = epsilon+weightDecayCoeff
  #  else:
  #    damping = tf.pow(epsilon+weightDecayCoeff, 1./num_factors)
  #  eigVals_tnorm_avg = [tf.reduce_mean(tf.abs(e)) for e in eigVals]
  #  for e, e_tnorm in zip(eigVals, eigVals_tnorm_avg):
  #    eig_tnorm_negList = [item for item in eigVals_tnorm_avg if item != e_tnorm]
  #    if len(eigVals) == 1:
  #      adjustment = 1. 
  #    elif len(eigVals) == 2:
  #      adjustment = tf.sqrt(e_tnorm / eig_tnorm_negList[0]) 
  #    else:
  #      eig_tnorm_negList_prod = reduce(lambda x,y: x*y, eig_tnorm_negList) 
  #      adjustment = tf.pow(tf.pow(e_tnorm, num_factors -1.) / eig_tnorm_negList_prod, 1./num_factors) 
  #    coeffs *= (e + adjustment*damping)
  #else:
  if not factored_damping or len(eigVals) == 1:
    damping = (epsilon + weightDecayCoeff)

    if use_KroneckerSum_dict.has_key(var):
        if use_KroneckerSum_dict[var] == 'option2':
            print 'use option2 for %s'%(var.name)
            coeffs = 1. - coeffs
            #coeffs = tf.Print(coeffs, [coeffs])
            coeffs += damping
        elif use_KroneckerSum_dict[var] == 'option1':
            print 'use option1 for %s'%(var.name)
            bstats_list = factor['bpropFactors']
            seqLength = tf.to_float(len(bstats_list))
            coeffs = rnn_geometric_sum(coeffs, seqLength)
            #coeffs += damping
        else:
            print 'use jimmys option for %s'%(var.name)
            print 'not using proper damping for %s'%(var.name)
            coeffs += damping

    else:
        coeffs += damping

    #coeffs = tf.Print(coeffs, [coeffs, tf.reduce_mean(coeffs)])
  return coeffs


def compute_whiten_tensors_eigen(factor, 
                                var, stats_eigens, 
                                damping_coeff, config,
                                use_KroneckerSum_dict=None):
    
    factored_damping=config['factored_damping']
    temporal_dependency=config['temporal_dependency']
    temporal_dependency_option=config['temporal_dependency_option']
    stats_names = ['bpropFactors_concat_stats_',
                   'fpropFactors_concat_stats_',]
    stats_types = ['covar', 'xcovar']
    inverse = {} 
    ## house-keeping: consolidate the eigen tensors
    whiten_matrices = {'covar':{}, 'xcovar':{}}
    stats_matrices = {'covar':[], 'xcovar':[]}
    for stats_type in stats_types:
      for stats_name in stats_names: 
          if factor.has_key(stats_name + stats_type + '_slot'):
              stats_slot = factor[stats_name + stats_type + '_slot']
              stats_matrices[stats_type].append(stats_slot)
              whiten_matrices[stats_type][stats_name] = {'Q': stats_eigens[stats_slot]['Q'], 'e': stats_eigens[stats_slot]['e']}
    ##
    ########
    if factored_damping:
      if len(whiten_matrices['covar']) == 2:
        print 'use factored damping for %s'%(var.name)
        print '#############'
        whiten_matrices['covar'] = dampFactors(whiten_matrices['covar'], damping_coeff)
    ########

    if temporal_dependency and len(whiten_matrices['covar']) == 2 and len(whiten_matrices['xcovar']) == 2: ## only works for two kronecker factors
      ## temporal correlation
      result_tensors = []
      print 'correct temporal dep for %s'%(var.name)
      cov_f, cov_b = stats_matrices['covar']
      xcov_f, xcov_b = stats_matrices['xcovar']

      if temporal_dependency_option == 'realoption2':
          (BB_e_f,BB_E_f,B_f, cov_inv_sqrt_f, 
           BB_e_b,BB_E_b,B_b, cov_inv_sqrt_b) = compute_rnn_whiten_tensor_option2_real(cov_f, xcov_f, 
                                    cov_b, xcov_b, 
                                    factor, stats, var,
                                    damping_coeff,   
                                    use_KroneckerSum_dict)
          result_tensors[0] = {'e':BB_e_b,'Q':BB_E_b,'B':B_b, 'cov_inv_sqrt':cov_inv_sqrt_b}
          result_tensors[1] = {'e':BB_e_f,'Q':BB_E_f,'B':B_f, 'cov_inv_sqrt':cov_inv_sqrt_f}
      else:
          if temporal_dependency_option == 'option1':
            rnn_whiten_func = compute_rnn_whiten_tensor_option1
          elif temporal_dependency_option == 'option2':
            rnn_whiten_func = compute_rnn_whiten_tensor_option2
          else:
            rnn_whiten_func = compute_rnn_whiten_tensor

          e_f, Q_f, e_b, Q_b = rnn_whiten_func(cov_f, xcov_f, 
                                                 cov_b, xcov_b, 
                                                 factor, stats, var,
                                                 damping_coeff,   
                                                 use_KroneckerSum_dict)
          result_tensors[0] = {'e':e_b,'Q':E_b,}
          result_tensors[1] = {'e':e_f,'Q':E_f,}
      inverse = {stats_name+'inverse':result_tensors[i] for i, stats_name in enumerate(stats_names)}
    else: 
      ## consolidate the eigen tensors
      for stats_name in stats_names: 
          if whiten_matrices['covar'].has_key(stats_name):
            inverse[stats_name + 'inverse'] = whiten_matrices['covar'][stats_name]
    return inverse

