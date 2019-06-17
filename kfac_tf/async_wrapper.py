import tensorflow as tf
import numpy as np
import kfac
from func_utils import *

class AKfacOptimizer(kfac.KfacOptimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, clip_kl=0.01, epsilon=1e-2, stats_decay=0.95, kfac_update = 20, 
                 stats_accum_iter = 60, full_stats_init =False, cold_iter = 100, cold_lr=None, 
                 async = False, async_stats = False, blockdiag_bias = False, 
                 channel_fac = False, factored_damping = True, temporal_dependency=False, temporal_dependency_option='option3',
                 approxT2 = False, use_float64 = False, use_diag = True, weight_decay_dict = {}):
        args_dict = dict(locals())
        del args_dict['self']
        super(AKfacOptimizer, self).__init__(**args_dict)

        self.inverse_queue_runner = None
        self.inverse_queue = None
        self.stats_queue_runner = None
        self.stats_queue = None

    def average_gradients(self, tower_grads):
      """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)
    
          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)
    
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
    
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
      return average_grads



    def average_stats(self, stats_list):
        stats_keys = set([key for i in stats_list for key in i.keys()])
        counter = {i:0 for i in stats_keys}
        avg_stats = {i:0 for i in stats_keys}
        for stats in stats_list:
            for stats_name, stats_tensor in stats.items():
                avg_stats[stats_name] += stats_tensor
                counter[stats_name] += 1
        for stats_name in avg_stats:
            avg_stats[stats_name] /= counter[stats_name]
        return avg_stats

    def compute_and_apply_inverse(self, inverse_dummy_op=None):
        def no_op_wrapper():
          return tf.no_op()

        args = [self.compute_inverses,
                self.apply_inverses, 
                self.stats_step,
                self._kfac_update,
                self._stats_accum_iter,
                no_op_wrapper,
                inverse_dummy_op]

        if self._async:
          # asynchronous eigen-decomp updates using queue
          updateInverseOps, self.inverse_queue, \
            self.inverse_queue_runner = queue_op_and_apply_results(*args)
          
        else:
          # synchronous eigen-decomp updates
          updateInverseOps = compute_op_and_apply_results(*args)
        return updateInverseOps

    def apply_stats_and_gradients(self, stats, grads):
        updateOps = []
        update_stats_op = self.apply_stats(stats)
        updateOps.append(update_stats_op)
        inverse_dummy_op = self.compute_inverses()

        dependency_list = []  if self._async else [update_stats_op]
        with tf.control_dependencies(dependency_list): 
          # compute inverses
          update_inverse_op = self.compute_and_apply_inverse(inverse_dummy_op)
          updateOps.append(update_inverse_op)
          with tf.control_dependencies([update_inverse_op]): 
            # compute updates
            updateParamsOps = self.apply_gradients(grads)
            updateOps.append(updateParamsOps)
        return tf.group(*updateOps), self.inverse_queue_runner

    def minimize(self, loss, loss_sampled, var_list=None):
        updateOps = []
        grads = self.compute_gradients(loss, var_list=var_list)
        # compute stats
        stats = self.compute_stats(loss_sampled, var_list = var_list)
        return self.apply_stats_and_gradients(stats, grads)

