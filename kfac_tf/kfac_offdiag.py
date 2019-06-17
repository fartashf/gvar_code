import tensorflow as tf
import numpy as np
import kfac, async_wrapper
from kfac_helper import *
from kfac_utils import *
from func_utils import *


class AKfacOptimizer(async_wrapper.AKfacOptimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, clip_kl=0.01, epsilon=1e-2, stats_decay=0.95, kfac_update = 20, 
                 stats_accum_iter = 60, full_stats_init =False, cold_iter = 100, cold_lr=None, 
                 async = False, async_stats = False, blockdiag_bias = False, 
                 channel_fac = False, factored_damping = True, temporal_dependency=False, temporal_dependency_option='option3',
                 approxT2 = False, use_float64 = False, use_diag = True, weight_decay_dict = {}, group_params = None):
        args_dict = dict(locals())
        del args_dict['self']
        del args_dict['group_params']
        super(AKfacOptimizer, self).__init__(**args_dict)
        self.group_params = group_params
        self.vFv = {params:tf.get_variable(name='KFAC/vFv_{}'.format(i), initializer=0., trainable=False) for i, params in enumerate(group_params)}

    def computeKfacPrecondUpdates(self, grads):
        gradlist, varlist = zip(*grads)
        
        updatelist = []
        vg = 0.

        assert len(self.factors)>0
        counter = 0

        factors = self.factors
        grad_dict = {var:grad for grad, var in zip(gradlist, varlist)}

        for grad, var in zip(gradlist, varlist):
          ## apply natural gradient if there are covariance matrix 
          ## but skip biases when using homogeneous coordinates
          whitenGradient = (factors[var].has_key('fpropFactors_concat_stats_covar')) \
                    or     (factors[var].has_key('bpropFactors_concat_stats_covar')) \
                    and    ((factors[var]['assnWeights'] is None) or self._config['blockdiag_bias'])

          if whitenGradient:
              counter += 1
              
              ## reshape gradient tensor              
              grad = reshapeGradients(grad, var, factors[var], 
                                      self._config)

              ## combine weights and biases
              if (factors[var]['assnBias'] is not None) and not self._config['blockdiag_bias']:
                ### use homogeneous coordinates only works for 2D grad. 
                ### TO-DO: figure out how to factorize bias grad
                ## stack bias grad
                var_assnBias = factors[var]['assnBias']
                grad = tf.concat(axis=0, values=[grad, tf.expand_dims(grad_dict[var_assnBias],0)])

              if self._config['temporal_dependency'] and self._config['temporal_dependency_option'] == 'realoption2' and self.factors[var]['KFACType'] == 'RNN':
                print 'apply realoption2 for %s'%(var.name)
                stats_f = self.stats[var]['fprop_concat_stats'][0]
                stats_b = self.stats[var]['bprop_concat_stats'][0]
                #print(self.stats_eigen[stats_f])
                cov_inv_sqrt_f = self.stats_eigen[stats_f]['cov_inv_sqrt']
                cov_inv_sqrt_b = self.stats_eigen[stats_b]['cov_inv_sqrt']
                B_f = self.stats_eigen[stats_f]['B']
                B_b = self.stats_eigen[stats_b]['B']
                ## transform into the whiten space
                grad = transform_grad(grad, cov_inv_sqrt_f, cov_inv_sqrt_b)
                grad = grad - transform_grad(grad, B_f, B_b, transpose_f = True)

                ##################################
                ## transform gradient to eigen space and reshape the eigenvalues for broadcasting
                grad, eigVals = transform_to_eigenspace(grad, var, factors[var])
                #grad, eigVals = transform_to_eigenspace2(grad, var, self.stats_eigen, self.stats[var])
                ##
                #####
                ## whiten using eigenvalues
                weightDecayCoeff = 0.
                if self._weight_decay_dict.has_key(var):
                   weightDecayCoeff = self._weight_decay_dict[var]
                   if KFAC_DEBUG:
                      print 'weight decay coeff for %s is %f'%(var.name, weightDecayCoeff)

                coeffs = compute_normalization_coeff(eigVals, var, self._epsilon, 
                                                     weightDecayCoeff, 
                                                     self._factored_damping,
                                                     self.factors[var],
                                                     self._use_KroneckerSum_dict)
                grad /= coeffs
                ##### 
                ## transform gradient back to euclidean space
                #grad, _ = transform_to_eigenspace2(grad, var, self.stats_eigen, self.stats[var], reverse=True)
                grad, _ = transform_to_eigenspace(grad, var, factors[var], reverse=True)
                ##################################
                grad = grad - transform_grad(grad, B_f, B_b, transpose_f = True, reverse=True)
                grad = transform_grad(grad, cov_inv_sqrt_f, cov_inv_sqrt_b)
                
              else:
                ## transform gradient to eigen space and reshape the eigenvalues for broadcasting
                print(var.name)
                grad, eigVals = transform_to_eigenspace(grad, var, factors[var])
                #grad, eigVals = transform_to_eigenspace2(grad, var, self.stats_eigen, self.stats[var])
                ##
                #####
                ## whiten using eigenvalues
                weightDecayCoeff = 0.
                if self._weight_decay_dict.has_key(var):
                   weightDecayCoeff = self._weight_decay_dict[var]
                   if KFAC_DEBUG:
                      print 'weight decay coeff for %s is %f'%(var.name, weightDecayCoeff)

                coeffs = compute_normalization_coeff(eigVals, var, self._epsilon, 
                                                     weightDecayCoeff, 
                                                     self._config['factored_damping'],
                                                     self.factors[var],
                                                     self._use_KroneckerSum_dict)
                grad /= coeffs
                ##### 
                ## transform gradient back to euclidean space
                grad, _ = transform_to_eigenspace(grad, var, factors[var], reverse=True)
                #grad, _ = transform_to_eigenspace2(grad, var, self.stats_eigen, self.stats[var], reverse=True)
                ##


              #grad = tf.Print(grad, [tf.convert_to_tensor('3'), tf.convert_to_tensor(var.name), grad.get_shape()])

              ## seperate weights and biases
              if (factors[var]['assnBias'] is not None) and not self._config['blockdiag_bias']:
                ### use homogeneous coordinates only works for 2D grad. 
                ### TO-DO: figure out how to factorize bias grad
                ## un-stack bias grad
                var_assnBias = factors[var]['assnBias']
                C_plus_one = int(grad.get_shape()[0])
                grad_assnBias = tf.reshape(tf.slice(grad,
                         begin=[C_plus_one-1, 0],
                         size=[1, -1]), var_assnBias.get_shape())
                grad_assnWeights = tf.slice(grad,
                         begin=[0, 0],
                         size=[C_plus_one-1, -1])
                grad_dict[var_assnBias] = grad_assnBias
                grad = grad_assnWeights

              ## reshape gradient tensor back
              grad = reshapeGradients(grad, var, factors[var], 
                                      self._config)
              ##

              
              grad_dict[var] = grad
        print('projecting %d gradient matrices'%counter)
        #updatelist = [grad_dict[var] for var in varlist]
        return grad_dict

        
    def getKfacPrecondUpdates(self, grads):
        gradlist, varlist = zip(*grads)
        grad_dict = {var:g for g, var in  grads}
        updates_dict = self.computeKfacPrecondUpdates(grads)
        vFv_list = [] 
        for params in self.group_params:
          vFv = 0.
          num_counter = 0
          for var in params:  
            g = grad_dict[var]
            num_counter += np.prod(g.get_shape().as_list())
            v = updates_dict[var]
            vFv += tf.reduce_sum(v*g)
            #local_vg = tf.reduce_sum(grad*g*(self._lr*self._lr))
          print("this group has {} number of paramters".format(num_counter))
          vFv_list.append(vFv)
        
        ## assume there are only two groups
        assert len(vFv_list) == 2, "there are more than two groups of params"
        alpha2 = vFv_list[0]*vFv_list[1]
        #lr_adjustments = [(1.-vFv_list[1])/(1.-alpha2), (1.-vFv_list[0])/(1.-alpha2)]
        #lr_adjustments = [(vFv_list[1])/(alpha2), (vFv_list[0])/(alpha2)]
        lr_adjustments = [(1.+vFv_list[1])/(1.+alpha2), (1.+vFv_list[0])/(1.+alpha2)]
        ### bound the lr
        lr_adjustments = [tf.minimum(lr_adj, 1.)  for lr_adj in lr_adjustments]
        lr_adjustments[1] = lr_adjustments[1]
        ###

        ## debug
        print_msg = [[tf.convert_to_tensor('lr adjust_{}: '.format(i)), lr_adj, tf.convert_to_tensor(' vFv: '), vFv_list[i]] for i, lr_adj in enumerate(lr_adjustments)]
        print_msg = [j for i in print_msg for j in i]
        print_msg.extend([tf.convert_to_tensor(' inv_steps: '), self.factor_step])
        lr_adjustments[0] = tf.Print(lr_adjustments[0], print_msg)
        ##


        assign_vFv_ops = [self.vFv[params].assign(vFv) for params, vFv in zip(self.group_params, vFv_list)]

        ### recale everything
        #scaling = tf.minimum(1., tf.sqrt(self._clip_kl/vg))
        with tf.control_dependencies(assign_vFv_ops):
          for i, params in enumerate(self.group_params):
            for var in params:
              updates_dict[var] *= lr_adjustments[i]

        updatelist = [updates_dict[var] for var in varlist]
        return zip(updatelist, varlist)

    def compute_and_apply_kfac_updates(self, grads):
        ##########
        #coldOptim = tf.train.MomentumOptimizer(self._cold_lr * (1. - self._momentum), self._momentum)
        #coldOptim = tf.train.AdamOptimizer(self._cold_lr * (1. - self._momentum))
        #coldOptim = tf.train.AdamOptimizer(self._cold_lr, beta1=self._momentum)
        coldOptim = tf.train.AdamOptimizer(self._cold_lr, beta1=0.5, beta2=0.9)
        sgd_optim_fn = lambda:coldOptim.apply_gradients(grads)
        ##########


        ########## KFAC
        optim = tf.train.MomentumOptimizer(self._lr * (1. - self._momentum), self._momentum)
        kfac_update_fn = lambda: cond_zipped_list_output(
                              tf.greater(self.factor_step, 
                                         tf.convert_to_tensor(0)), 
                              grads,
                      lambda:self.getKfacPrecondUpdates(grads), 
                      lambda:[(tf.zeros_like(g), v) for g, v in grads])
                      #lambda:grads)
        kfac_optim_fn = lambda:optim.apply_gradients(kfac_update_fn())
        ##########
        return scheduledOptimOp(sgd_optim_fn, kfac_optim_fn, 
                                self.sgd_step, self._cold_iter)
 
