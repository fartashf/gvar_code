import tensorflow as tf
import numpy as np
from kfac_helper import *
from kfac_utils import *
from func_utils import *
import kfac_search


class KfacOptimizer(object):
      
    def __init__(self, learning_rate=0.01, momentum=0.9, clip_kl=0.01, epsilon=1e-2, stats_decay=0.95, kfac_update = 20, 
                 stats_accum_iter = 60, full_stats_init =False, cold_iter = 100, cold_lr=None, 
                 async = False, async_stats = False, blockdiag_bias = False, 
                 channel_fac = False, factored_damping = True, temporal_dependency=False, temporal_dependency_option='option3',
                 approxT2 = False, use_float64 = False, use_diag = True, weight_decay_dict = {}):
        ###### optimizer hyper paramters
        self._lr = learning_rate
        self._momentum = momentum
        self._clip_kl = clip_kl
        self._kfac_update = kfac_update
        self._async = async
        self._async_stats = async_stats
        self._epsilon = epsilon
        self._stats_decay = stats_decay
        
        ###### additional config flags
        self._config = {}
        self._config['blockdiag_bias'] = blockdiag_bias
        self._config['useHomogeneous'] = (not blockdiag_bias)
        self._config['channel_fac'] = channel_fac
        self._config['approxT2'] = approxT2
        self._config['use_diag'] = use_diag
        self._config['factored_damping'] = factored_damping
        self._config['temporal_dependency'] = temporal_dependency
        self._config['temporal_dependency_option'] = temporal_dependency_option
        self._config['use_float64'] = use_float64
        ###### 

        self._cold_iter = cold_iter
        if cold_lr == None:
            ## good heuristics
            self._cold_lr = self._lr * 3.
        else:
            self._cold_lr = cold_lr
        self._stats_accum_iter = stats_accum_iter
        self._weight_decay_dict = weight_decay_dict
        self._diag_init_coeff = 0.
        self._full_stats_init = full_stats_init
        if not self._full_stats_init:
            self._stats_accum_iter = self._cold_iter

        self.sgd_step = tf.Variable(0, name='KFAC/sgd_step', trainable=False)
        self.global_step = tf.Variable(0, name='KFAC/global_step', trainable=False)
        self.factor_step = tf.Variable(0, name='KFAC/factor_step', trainable=False)
        self.stats_step = tf.Variable(0, name='KFAC/stats_step', trainable=False)
        self.vFv = tf.Variable(0., name='KFAC/vFv', trainable=False)
        
        self.factors = {}
        self.param_vars = []
        self.stats = {}
        self.stats_eigen = {}
        self.stats_slot_list = []
        self._stats_decay_rnn = {}
        self._use_KroneckerSum_dict = {}



    def getFactors(self, g, varlist):
        graph = tf.get_default_graph() 
        factorTensors = {}
        fpropTensors = []
        bpropTensors = []
        fops = []

        for t, param in zip(g, varlist):
            if KFAC_DEBUG:
                print 'get factor for ', param.name, param.get_shape()
            factors = kfac_search.searchFactors(t, param, graph)
            factorTensors[param] = factors

            #if factors['opName'] not in KFAC_OPS:
            #  assert len(factors['bpropFactors'][0].get_shape()) == 2, "%s is not supported, Unknown OP only works with rank-2 factors"%(factors['opName'])
        
        ## check variable op types and approximation stratergy
        factorTensors = kfac_search.searchApproxStratergy(factorTensors, varlist, self._config)
        ########

        if KFAC_DEBUG:
          #print(factorTensors)
          print '******* Factors: '
          for items in varlist:
            print items.name
            if factorTensors[items]['fpropFactors_concat'] is not None:
              print "forward factor name:\t", factorTensors[items]['fpropFactors_concat'].name
              print "forward factor shape:\t", factorTensors[items]['fpropFactors_concat'].get_shape()
            if factorTensors[items]['bpropFactors_concat'] is not None:
              print "backward factor name:\t", factorTensors[items]['bpropFactors_concat'].name
              print "backward factor shape:\t", factorTensors[items]['bpropFactors_concat'].get_shape()
            print "################"
          print '*******'
        ########
        
        ########
        ## construct graph for the actual sampled Fisher blocks
        factorTensors = getSampledFisher(factorTensors)
        ######## 

        self.factors = mergeFactors(self.factors, factorTensors)
        return self.factors



    def compute_and_apply_stats(self, loss_sampled, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()
        
        statsUpdates = self.compute_stats(loss_sampled, var_list = varlist)
        return self.apply_stats(statsUpdates) 


    def compute_stats(self, loss_sampled, var_list=None, create_slots=True):
        stats = self.compute_stats2(loss_sampled, var_list = var_list, create_slots = create_slots)
        statsUpdates = self.init_slots(self._diag_init_coeff, stats, self.factors)
        return statsUpdates


    def compute_stats2(self, loss_sampled, var_list=None, create_slots=True, full_diag_approx = False, gradients = None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()
        ### take the gradient w.r.t. sampled targets.
        if gradients is None:
          gs = tf.gradients(loss_sampled, varlist, name='gradientsSampled')
        else:
          gs = gradients
        self.gs = gs
        ### search for the Fisher blocks w.r.t. the variables
        factors = self.getFactors(gs, varlist)
        stats_results = {}
        ######
        factor_concat_keys = ['fpropFactors_concat', 'bpropFactors_concat']
        statsUpdates_cache = {'covar':{}, 'xcovar':{}}
        for var in factors:
          with tf.name_scope(tensor_name(var)) as scope:
            compute_stats_funcs = {'covar':compute_stats_default, }
            stats_results[var] = {}
            #### compute cross covariance for RNN approx
            if factors[var]['KFACType'] == 'RNN' and self._config['temporal_dependency']:
              compute_stats_funcs['xcovar'] = compute_xcovar_default
            for stats_name, compute_stats_func in compute_stats_funcs.items():
              if full_diag_approx:
                if factors[var][factor_concat_keys[0]] is not None:
                  factor_id = (factors[var][concat_tensor+'_id'] for concat_tensor in factor_concat_keys)
                else:
                  factor_id = factors[var][factor_concat_keys[-1]+'_id']
                concat_tensor = factor_concat_keys[-1] 
                with tf.name_scope(concat_tensor+'_stats_'+stats_name) as scope:
                      if not statsUpdates_cache[stats_name].has_key(factor_id):
                        cov = compute_stats_func(var, factors[var], 
                                                    True, True,
                                                    self._config,
                                                    self._stats_decay_rnn)
                        statsUpdates_cache[stats_name][factor_id] = cov
                      else:
                        cov = statsUpdates_cache[stats_name][factor_id]
                      result_keyname = 'diagFactors_tensor'+'_stats_'+stats_name
                      factors[var][result_keyname] = cov
                      stats_results[var][result_keyname] = cov

              else:
                #### compute stats for both forward and backward factors
                for concat_tensor, (fprop_flag, bprop_flag) in zip(factor_concat_keys,
                                                          [(True, False,), (False, True,)]):
                  if factors[var][concat_tensor] is not None:
                    
                    factor_id = factors[var][concat_tensor+'_id']
                    #### compute once if factors is shared across layers
                    with tf.name_scope(concat_tensor+'_stats_'+stats_name) as scope:
                      if not statsUpdates_cache[stats_name].has_key(factor_id):
                        cov = compute_stats_func(var, factors[var], 
                                                    fprop_flag, bprop_flag,
                                                    self._config,
                                                    self._stats_decay_rnn)
                        statsUpdates_cache[stats_name][factor_id] = cov
                      else:
                        cov = statsUpdates_cache[stats_name][factor_id]
                      factors[var][concat_tensor+'_stats_'+stats_name] = cov
                      stats_results[var][concat_tensor+'_stats_'+stats_name] = cov
        stats_results = {key:item for key, item in stats_results.items() if len(item)>0}
        return stats_results



    def updateAccumStats(self,statsUpdates):
        if self._full_stats_init:
            return tf.cond(tf.greater(self.sgd_step, self._cold_iter), lambda: tf.group(*self._apply_stats(statsUpdates, accumulate=True, accumulateCoeff=1./max(self._stats_accum_iter,1))), tf.no_op)
        else:
            return tf.group(*self._apply_stats(statsUpdates, accumulate=True, accumulateCoeff=1./max(self._stats_accum_iter,1)))

    def updateRunningAvgStats(self,statsUpdates, fac_iter=1):
        return tf.group(*self._apply_stats(statsUpdates))

    def apply_stats(self, statsUpdates):
        """ compute stats and update/apply the new stats to the running average
        """
        self.stats_slot_list = [tensor_slot for tensor_slot, tensor in statsUpdates.items()] ## obtain all the running average stats

        # synchronous stats update
        update_stats_op = tf.cond(tf.greater_equal(self.stats_step, self._stats_accum_iter), 
                                  lambda: self.updateRunningAvgStats(statsUpdates), 
                                  lambda: self.updateAccumStats(statsUpdates))

        self._update_stats_op = update_stats_op
        return update_stats_op

    def _apply_stats(self, statsUpdates, accumulate=False, accumulateCoeff=0.):
        updateOps = []
        # obtain the stats var list
        for stats_var, stats_new in statsUpdates.items():
            if accumulate:
                # simple superbatch averaging
                update_op = tf.assign_add(stats_var, accumulateCoeff * stats_new , use_locking=True)
            else:
                # exponential running averaging
                update_op, _ = updateRunningAverage(stats_new, stats_var, self._stats_decay, self.stats_step)
            updateOps.append(update_op)


        with tf.control_dependencies(updateOps):
          stats_step_op = tf.assign_add(self.stats_step, 1)
     
        if KFAC_DEBUG:
          print("=======update stats======")
          print(updateOps)
          stats_step_op = (tf.Print(stats_step_op, 
                [tf.convert_to_tensor('step:'),
                 self.global_step, 
                 tf.convert_to_tensor('fac step:'),
                 self.factor_step, 
                 tf.convert_to_tensor('sgd step:'),
                 self.sgd_step, 
                 tf.convert_to_tensor('Accum:'),
                 tf.convert_to_tensor(accumulate), 
                 tf.convert_to_tensor('Accum coeff:'),
                 tf.convert_to_tensor(accumulateCoeff),
                 tf.convert_to_tensor('stat step:'),
                 self.stats_step, updateOps[0], updateOps[1]]))
        return [stats_step_op,]


    def compute_and_apply_inverse(self):
        # synchronous eigen-decomp updates
        inverses = self.compute_stats_eigen(loss_sampled, var_list = varlist)
        inverseUpdates = self.init_inverse_slots(inverses)
        return self.apply_stats_eigen(inverseUpdates) 


    def _compute_eigen_decomp(self, stats_slots):
      """ compute the eigen decomp using copied var stats to avoid concurrent read/write from other queue """
      # TO-DO: figure out why this op has delays (possibly moving eigenvectors around?)
      eigens = {}
 
      ## compute   
      with tf.device('/cpu:0'):
        with tf.control_dependencies([]):
          for stats_var in stats_slots:
              if not eigens.has_key(stats_var):
                  
                  name = tensor_name(stats_var)+'/eig'
                  if len(stats_var.get_shape()) == 1:
                    e = tf.identity(stats_var + 1e-6, name=name)
                    Q = None
                  else:
                    damping = tf.diag(tf.ones((tf.shape(stats_var)[0],))*1e-6)
                    eig_results = tf.self_adjoint_eig(stats_var + damping,
                                                name = name)
                    e = eig_results[0]
                    Q = eig_results[1]
                    ## reset the shape if it somehow is missing.
                    if Q.get_shape()[0].value is None:
                      stats_shape = stats_var.get_shape()
                      Q.set_shape(stats_shape)
                      e.set_shape(stats_shape[:1])
                  eigens[stats_var] = {'e':e, 'Q':Q}

      return eigens


    def compute_inverses_eigen(self, stats_list = None):
      """ compute the eigen decomp using copied var stats to avoid concurrent read/write from other queue """
      # TO-DO: figure out why this op has delays (possibly moving eigenvectors around?)
      factors = self.factors
      inverses = {}
      if stats_list is None:
        stats_list = self.stats_slot_list
      ############### do eigen decomp ##########
      stats_eigens = self._compute_eigen_decomp(self.stats_slot_list)
      ##### compute the preconditioning/whitening(usually are the eigenvectors) matrices
      for var, factor in factors.items():
        inverse = compute_whiten_tensors_eigen(factor, 
                         var, stats_eigens,
                         damping_coeff=self._epsilon,
                         config=self._config, 
                         use_KroneckerSum_dict=self._use_KroneckerSum_dict)
        if len(inverse)>0:
          inverses[var] = inverse
      return inverses


    def compute_inverses(self, stats_list = None):
      inverses = self.compute_inverses_eigen(stats_list)
      inversesUpdates = self.init_slots(1., inverses, self.factors)
      return inversesUpdates


    def apply_inverses(self, eigenUpdates):
        updateOps = []
        print('updating %d eigenvalue/vectors'%len(eigenUpdates))
        for stats_eigen_var, tensor  in eigenUpdates.items():
            updateOps.append(tf.assign(stats_eigen_var, tensor, use_locking=True))

        with tf.control_dependencies(updateOps): 
          factor_step_op = tf.assign_add(self.factor_step, 1)
          updateOps.append(factor_step_op)
          if KFAC_DEBUG:
            updateOps.append(tf.Print(tf.constant(0.), [tf.convert_to_tensor('updated kfac eigen')]))
        return updateOps

    def compute_and_apply_inverse(self):
        def no_op_wrapper():
          return tf.no_op()

        args = [self.compute_inverses,
                self.apply_inverses, 
                self.stats_step,
                self._kfac_update,
                self._stats_accum_iter,
                no_op_wrapper,]

        # synchronous eigen-decomp updates
        updateInverseOps = compute_op_and_apply_results(*args)
        return updateInverseOps


    def init_slots(self, diag_coeff, tensor_dict, master_dict, slots_lookup = None):
      if slots_lookup is None:
        slots_lookup = {}
      for tensor_key, tensor in tensor_dict.items():
        ##########
        name = tensor_key if isinstance(tensor_key, str) else tensor_name(tensor_key)
        with tf.variable_scope(name) as scope:
          if isinstance(tensor, dict):
            if not master_dict.has_key(tensor_key):
              master_dict[tensor_key] = {} 
            self.init_slots(diag_coeff, tensor, master_dict[tensor_key], slots_lookup)
          else:
            if tensor is not None:
                if master_dict.has_key(tensor_key + '_slot'):
                  slot = master_dict[tensor_key + '_slot']
                else:
                  slot = createSlotLike(tensor, diag_coeff, slots_lookup)
                  master_dict[tensor_key + '_slot'] = slot
                slots_lookup[tensor] = slot
        ##########
      return {i:k for k, i in slots_lookup.items()}


    def getKfacPrecondUpdates(self, grads):
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
        
        for g, var in zip(gradlist, varlist):  
            grad = grad_dict[var]
            ### clipping ###
            if KFAC_DEBUG:
              print 'apply clipping to %s'%(var.name)
            
            local_vg = tf.reduce_sum(grad*g*(self._lr*self._lr))
            vg += local_vg
            
        
        ### recale everything
        if KFAC_DEBUG:
            print 'apply vFv clipping'
        
        scaling = tf.minimum(1., tf.sqrt(self._clip_kl/vg))
        if KFAC_DEBUG:
          scaling = tf.Print(scaling, [tf.convert_to_tensor('clip: '), scaling, tf.convert_to_tensor(' vFv: '), vg])
        with tf.control_dependencies([tf.assign(self.vFv, vg)]):
          updatelist = [grad_dict[var] for var in varlist]
          for i, item in enumerate(updatelist):
            updatelist[i] = scaling * item

        #return updatelist
        return zip(updatelist, varlist)

    def compute_gradients(self, loss, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()
        g = tf.gradients(loss, varlist)
       
        return [(a,b) for a,b in zip(g, varlist)]


    def compute_and_apply_kfac_updates(self, grads):
        ##########
        coldOptim = tf.train.MomentumOptimizer(self._cold_lr * (1. - self._momentum), self._momentum)
        #coldOptim = tf.train.AdamOptimizer(self._cold_lr * (1. - self._momentum))
        #coldOptim = tf.train.AdamOptimizer(self._cold_lr, beta1=self._momentum)
        sgd_optim_fn = lambda:coldOptim.apply_gradients(grads)
        ##########


        ########## KFAC
        optim = tf.train.MomentumOptimizer(self._lr * (1. - self._momentum), self._momentum)
        kfac_update_fn = lambda: cond_zipped_list_output(
                              tf.greater(self.factor_step, 
                                         tf.convert_to_tensor(0)), 
                              grads,
                      lambda:self.getKfacPrecondUpdates(grads), 
                      lambda:grads)
        kfac_optim_fn = lambda:optim.apply_gradients(kfac_update_fn())
        ##########
        return scheduledOptimOp(sgd_optim_fn, kfac_optim_fn, 
                                self.sgd_step, self._cold_iter)
        
    def apply_gradients(self, grads):
        global_step_op = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_op]):
          updateOps = self.compute_and_apply_kfac_updates(grads)
        return updateOps


    def minimize(self, loss, loss_sampled, var_list=None):
        updateOps = []
        grads = self.compute_gradients(loss, var_list=var_list)
        # compute stats
        update_stats_op = self.compute_and_apply_stats(loss_sampled, 
                                                       var_list=var_list)
        updateOps.append(update_stats_op)
        ## build graph first to init slots
        _ = self.compute_inverses()

        dependency_list = [update_stats_op]
        with tf.control_dependencies(dependency_list): 
          # compute inverses
          update_inverse_op = self.compute_and_apply_inverse()
          updateOps.append(update_inverse_op)
          with tf.control_dependencies([update_inverse_op]): 
            # compute updates
            updateParamsOps = self.apply_gradients(grads)
            updateOps.append(updateParamsOps)
        return tf.group(*updateOps), None


