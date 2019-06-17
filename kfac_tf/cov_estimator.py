import tensorflow as tf
import numpy as np
import kfac
from func_utils import *

class Estimator(kfac.KfacOptimizer):
    def __init__(self, epsilon=1e-3, blockdiag_bias = True, 
                 channel_fac = False, factored_damping = True, 
                 approx_option = 'kfac',
                 temporal_dependency=False, 
                 temporal_dependency_option='option3',
                 approxT2 = False, use_float64 = False, 
                 weight_decay_dict = {}):

        self._epsilon = epsilon
        
        ###### additional config flags
        self._config = {}
        self._config['approx_option'] = approx_option
        self._config['blockdiag_bias'] = blockdiag_bias
        self._config['useHomogeneous'] = (not blockdiag_bias)
        self._config['channel_fac'] = channel_fac
        self._config['approxT2'] = approxT2
        self._config['factored_damping'] = factored_damping
        self._config['temporal_dependency'] = temporal_dependency
        self._config['temporal_dependency_option'] = temporal_dependency_option
        self._config['use_float64'] = use_float64
        ###### 

        self.factors = {}
        self.param_vars = []
        self.stats = {}
        self.stats_eigen = {}
        self._stats_decay_rnn = {}
        self._use_KroneckerSum_dict = {}


    def compute_stats(self, loss_sampled, var_list=None, gradients=None):
        if var_list is None:
          var_list = tf.trainable_variables()

        if self._config['approx_option'] == 'exact':
          if len(loss_sampled.get_shape()) >= 1:
            self.loss_per_example = loss_sampled 
            raise NotImplementedError
          else:
            print(loss_sampled)
            print("Need loss per training example")
            raise NotImplementedError
        else:
          full_diag_approx = True if self._config['approx_option'] == 'diag' else False
          stats = self.compute_stats2(loss_sampled, var_list = var_list, full_diag_approx=full_diag_approx, gradients=gradients)
          self.stats = stats
          return stats

    def pow(self, A, power=0.5):
        if power == 0.5:
          L = tf.cholesky(A + 1e-6*tf.eye(int(A.get_shape()[-1])))
        else:
        #if True:
          A = A + 1e-8*tf.eye(int(A.get_shape()[-1]))
          e, Q = tf.self_adjoint_eig(A)
          ## clip e
          e = tf.maximum(e, 1e-8)
          ##
          L = tf.matmul(tf.matmul(Q, tf.diag(tf.pow(e, power))),Q, transpose_b=True)
        return L

    def fvp_kfac(self, stats, vec, power):
      if len(stats) == 2:
        vec_shape = vec.get_shape()
        if len(vec_shape) == 4:
          vec = tf.reshape(vec, (-1, int(vec_shape[-1])))
        #FVP 
        B = stats['bpropFactors_concat_stats_covar']
        A = stats['fpropFactors_concat_stats_covar']
        #A = tf.stop_gradient(A)
        #B = tf.stop_gradient(B)
        if power != 1.:
          #raise NotImplementedError 
          A = self.pow(A, power)
          B = self.pow(B, power)
        vec = tf.matmul(tf.matmul(A, vec), B)
 
        if len(vec_shape) == 4:
          vec = tf.reshape(vec, vec_shape)
      elif len(stats) == 1:
        ## MC block-approx
        B = stats['bpropFactors_concat_stats_covar']
        #B = tf.stop_gradient(B)
        if power != 1.:
          B = self.pow(B, power)
          #raise NotImplementedError 
        if len(B.get_shape()) == 1:
          ## diagonal factor
          vec *= B
        else:
          vec = tf.squeeze(tf.matmul(tf.expand_dims(vec,0), B))
      else:
        raise NotImplementedError
      return vec

    def fvp_diag(self, stats, vec, power):
      if len(stats) == 1:
        ## full diagonal-approx
        #print('full diag approx')
        #print(stats)
        B = stats.values()[-1]
        if power != 1.:
          B = tf.pow(B, power)
        vec *= B
      else:
        raise NotImplementedError
      return vec

    def fvp_exact(self, var, vec, power):
      if power == 1.:
        ##########################
        # Fisher Vector Product 
        def fwd_gradients(ys, xs, d_xs):
          """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
                  With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
                  the vector being pushed forward."""
          v = tf.ones_like(ys)  # dummy variable
          g = tf.gradients(ys, xs, grad_ys=v)
          return tf.gradients(g, v, grad_ys=d_xs)
        per_data_point_loss = self.loss_per_example
        Jv = fwd_gradients(per_data_point_loss \
                          /tf.to_float(tf.shape(per_data_point_loss)[0]), 
                           var, vec) #jacobian vector product
        Fv = tf.gradients(per_data_point_loss, var, grad_ys=Jv) #fisher vector product
        return Fv
      else:
        raise NotImplementedError
        

    def fvp(self, stats, vec, power):
        if self._config['approx_option'] == 'kfac':
          vec = self.fvp_kfac(stats, vec, power)
        elif self._config['approx_option'] == 'diag':
          vec = self.fvp_diag(stats, vec, power)
        else:
          raise NotImplementedError 

        return vec

    def FisherVectorProduct(self, var_vec_list, power=1.):
        var, vec = zip(*var_vec_list)
        result_list = []
        if self._config['approx_option'] == 'exact':
          return self.fvp_exact(var, vec, power)
        else:
          for var, vec in var_vec_list:
            result_list.append(self.fvp(self.stats[var], vec, power))
        return result_list

