# Modified Adam Optimizer

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Adam for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
import pdb

class AdamNoisyOptimizer(tf.train.AdamOptimizer):

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam"):
    super(AdamNoisyOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._beta1_power = None
    self._beta2_power = None

    # Created in SparseApply if needed.
    self._updated_lr = None

  # def _apply_sparse(self, grad, noisy_grad, var):
  def _apply_sparse(self, grad, noise, var):
    lr = (self._lr_t *
          math_ops.sqrt(1 - self._beta2_power)
          / (1 - self._beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - self._beta1_t)
    m_t = state_ops.assign(m, m * self._beta1_t,
                           use_locking=self._use_locking)
    m_t = state_ops.assign_add(m_t, m_scaled_g_values,
                               use_locking=self._use_locking)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    # v_scale_half = tf.multiply(grad,grad)
    v_scaled_g_values = tf.multiply(grad,grad) * (1 - self._beta2_t)
    v_t = state_ops.assign(v, v * self._beta2_t, use_locking=self._use_locking)
    v_t = state_ops.assign_add(v_t, v_scaled_g_values,
                               use_locking=self._use_locking)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(var,
                                     # lr * m_t / (v_sqrt + self._epsilon_t) - noise,
                                      tf.clip_by_global_norm(lr * m_t / (v_sqrt + self._epsilon_t) - noise, 10.)[0],
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def apply_gradients(self, grads_and_vars, noise, global_step=None, name=None):
    # This is a default implementation of apply_gradients() that can be shared
    # by most optimizers.  It relies on the subclass implementing the following
    # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works
    for g, v in grads_and_vars:
      if not isinstance(g, (ops.Tensor, ops.IndexedSlices, type(None))):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      if not isinstance(v, variables.Variable):
        raise TypeError(
            "Variable must be a tf.Variable: %s" % v)
      if g is not None:
        self._assert_valid_dtypes([g, v])
    var_list = [v for g, v in grads_and_vars if g is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s" %
                       (grads_and_vars,))
    with ops.control_dependencies(None):
      self._create_slots(var_list)
    update_ops = []
    with ops.op_scope([], name, self._name) as name:
      self._prepare()
      for (grad, var), n in zip(grads_and_vars, noise):
        if grad is None:
          continue
        with ops.name_scope("update_" + var.op.name), ops.device(var.device):
          update_ops.append(self._apply_sparse(grad, n, var))
      if global_step is None:
        return self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.colocate_with(global_step):
            return state_ops.assign_add(global_step, 1, name=name).op


  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._beta1_power):
        update_beta1 = self._beta1_power.assign(
            self._beta1_power * self._beta1_t,
            use_locking=self._use_locking)
        update_beta2 = self._beta2_power.assign(
            self._beta2_power * self._beta2_t,
            use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                  name=name_scope)



# # Another version?
# # Copyright 2015 Google Inc. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

# """Adam for TensorFlow."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import state_ops
# from tensorflow.python.ops import variables
# from tensorflow.python.training import optimizer
# from tensorflow.python.training import training_ops


# class AdamNoisyOptimizer(tf.train.AdamOptimizer):

#   def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
#                use_locking=False, name="Adam"):
#     super(AdamNoisyOptimizer, self).__init__(use_locking, name)
#     self._lr = learning_rate
#     self._beta1 = beta1
#     self._beta2 = beta2
#     self._epsilon = epsilon

#     # Tensor versions of the constructor arguments, created in _prepare().
#     self._lr_t = None
#     self._beta1_t = None
#     self._beta2_t = None
#     self._epsilon_t = None

#     # Variables to accumulate the powers of the beta parameters.
#     # Created in _create_slots when we know the variables to optimize.
#     self._beta1_power = None
#     self._beta2_power = None

#     # Created in SparseApply if needed.
#     self._updated_lr = None

#   def _apply_sparse(self, grad, noisy_grad, var):
#     lr = (self._lr_t *
#           math_ops.sqrt(1 - self._beta2_power)
#           / (1 - self._beta1_power))
#     # m_t = beta1 * m + (1 - beta1) * g_t
#     m = self.get_slot(var, "m")
#     m_scaled_g_values = noisy_grad * (1 - self._beta1_t)
#     m_t = state_ops.assign(m, m * self._beta1_t,
#                            use_locking=self._use_locking)
#     m_t = state_ops.assign_add(m_t, m_scaled_g_values,
#                                use_locking=self._use_locking)
#     # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
#     v = self.get_slot(var, "v")
#     v_scaled_g_values = (grad * grad) * (1 - self._beta2_t)
#     v_t = state_ops.assign(v, v * self._beta2_t, use_locking=self._use_locking)
#     v_t = state_ops.assign_add(v_t, v_scaled_g_values,
#                                use_locking=self._use_locking)
#     v_sqrt = math_ops.sqrt(v_t)
#     var_update = state_ops.assign_sub(var,
#                                       lr * m_t / (v_sqrt + self._epsilon_t),
#                                       use_locking=self._use_locking)
#     return control_flow_ops.group(*[var_update, m_t, v_t])

#   def apply_gradients(self, grads_and_vars, noisy_grads, global_step=None, name=None):
#     # This is a default implementation of apply_gradients() that can be shared
#     # by most optimizers.  It relies on the subclass implementing the following
#     # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().
#     grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works
#     for g, v in grads_and_vars:
#       if not isinstance(g, (ops.Tensor, ops.IndexedSlices, type(None))):
#         raise TypeError(
#             "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
#       if not isinstance(v, variables.Variable):
#         raise TypeError(
#             "Variable must be a tf.Variable: %s" % v)
#       if g is not None:
#         self._assert_valid_dtypes([g, v])
#     var_list = [v for g, v in grads_and_vars if g is not None]
#     if not var_list:
#       raise ValueError("No gradients provided for any variable: %s" %
#                        (grads_and_vars,))
#     with ops.control_dependencies(None):
#       self._create_slots(var_list)
#     update_ops = []
#     with ops.op_scope([], name, self._name) as name:
#       self._prepare()
#       for (grad, var), noisy_grad in zip(grads_and_vars, noisy_grads):
#         if grad is None:
#           continue
#         with ops.name_scope("update_" + var.op.name), ops.device(var.device):
#           update_ops.append(self._apply_sparse(grad, noisy_grad, var))
#       if global_step is None:
#         return self._finish(update_ops, name)
#       else:
#         with ops.control_dependencies([self._finish(update_ops, "update")]):
#           with ops.colocate_with(global_step):
#             return state_ops.assign_add(global_step, 1, name=name).op


#   def _finish(self, update_ops, name_scope):
#     # Update the power accumulators.
#     with ops.control_dependencies(update_ops):
#       with ops.colocate_with(self._beta1_power):
#         update_beta1 = self._beta1_power.assign(
#             self._beta1_power * self._beta1_t,
#             use_locking=self._use_locking)
#         update_beta2 = self._beta2_power.assign(
#             self._beta2_power * self._beta2_t,
#             use_locking=self._use_locking)
#     return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
#                                   name=name_scope)
