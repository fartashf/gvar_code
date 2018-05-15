import tensorflow as tf
import numpy as np   
from collections import OrderedDict

def flatten_dict_ops(dict_ops):
    op_list = []
    for key, item in dict_ops.items():
      if isinstance(item, dict):
        op_list+=flatten_dict_ops(item)
      else:
        op_list.append(item)
    return op_list

def tf_x_mod_y_eq_z(x,y,z = None):
  if z is None:
    z = tf.convert_to_tensor(0)
  return tf.equal(tf.mod(x, y), z)

def refresh_condition(counter, k, n):
  return tf.logical_and(
            tf_x_mod_y_eq_z(counter, k),
            tf.greater_equal(counter, n))

def pushOpToQueue(op_fn, counter, k, n, dummy_op = None):
        # push op_fn to a queue thread if ( mod(counter, k) == 0 and counter > n )

        print 'Use async queue'
        # get the list of resulting tensors from computeStatsEigen
        if dummy_op is None:
          dummy_op = op_fn()
        slot_list, result_list = zip(*dummy_op.items())
        def op_fn_flatten():
           op_dict = op_fn()
           return [op_dict[slot] for slot in slot_list]
        #op_fn_flatten = lambda : [result for slot, result in op_fn().items()]
        deflatten_dequeue_op = lambda : {slot:result for slot, result in zip(slot_list, queue.dequeue())}
        # define a queue for the list of factor loading tensors 
        queue = tf.FIFOQueue(1, [result.dtype for result in result_list], 
                              shapes = [result.get_shape() for result in result_list])

        enqueue_op = tf.cond( 
                      refresh_condition(counter, k, n),
                      lambda: queue.enqueue(op_fn_flatten()), 
                      tf.no_op)
        ## enqueue_op blocks so we are running it on a seperate QueueRunner thread and 
        ## the results will be collected by the queue
        queue_runner = tf.train.QueueRunner(queue, [enqueue_op])
        return deflatten_dequeue_op, queue, queue_runner

def queue_is_empty(queue):
        return tf.equal(queue.size(), tf.convert_to_tensor(0))

def queue_op_and_apply_results(op_fn, apply_fn, counter, k, n, no_op_wrapper, dummy_op=None):
        dequeue_op_fn, queue, queue_runner = pushOpToQueue(op_fn, counter, k, n, dummy_op)

        apply_op = tf.cond(tf.greater_equal(counter, n), 
                      lambda:tf.cond(queue_is_empty(queue),
                                     tf.no_op, 
                                     lambda:tf.group(*apply_fn(dequeue_op_fn()))),
                      no_op_wrapper)
        return apply_op, queue, queue_runner

def compute_op_and_apply_results(op_fn, apply_fn, counter, k, n, no_op_wrapper, dummy_op=None):
        # run apply_fn if ( mod(counter, k) == 0 and counter > n )
        apply_op = tf.cond(
                      refresh_condition(counter, k, n),
                      lambda: tf.group(*apply_fn(op_fn())), 
                      no_op_wrapper)
        #apply_op = tf.group(*apply_fn(op_fn))
        return apply_op

def cond_zipped_list_output(pred, zipped_list, fn_1, fn_2):
        v1, v2 = zip(*zipped_list)
        fn_1_unpack = lambda: zip(*fn_1())[0]
        fn_2_unpack = lambda: zip(*fn_2())[0]
        return zip(tf.cond(pred, fn_1_unpack, fn_2_unpack), v2)

def scheduledOptimOp(optim1_fn, optim2_fn, counter, k, debug = False):
        def optim1_update():
          step_op = tf.assign_add(counter, 1)
          optim1_op = optim1_fn()
          if debug:
            with tf.control_dependencies([step_op, coldOptim_op]):
              sgd_step_op = tf.Print(step_op, [counter, tf.convert_to_tensor('doing cold step')])
          return tf.group(*[step_op, optim1_op])

        optim_op = tf.cond(tf.greater(counter, k),
                           optim2_fn,
                           optim1_update)
        return optim_op
