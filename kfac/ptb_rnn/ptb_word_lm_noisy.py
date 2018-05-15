# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import time

import numpy as np
import tensorflow as tf
import reader
import util
#from tensorflow.models.rnn.ptb import reader
import rnn_cell
import reader
import cov_estimator

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "data",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("saved_path", None, "Output of the log file.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_boolean('useSGD', False,"""use SGD optimizer""")
flags.DEFINE_boolean('useAdam', False,"""use Adam optimizer""")
flags.DEFINE_boolean('useSGD_diagF', False, """use SGD optimizer with diagonal
    Fisher noise""")
flags.DEFINE_boolean('useAdam_diagF', False, """use Adam optimizer with diagonal
    Fisher noise""")
flags.DEFINE_float('lr', 1.0,"""learning rate""")
flags.DEFINE_boolean('debug', False,"""turn on debug flag""")
flags.DEFINE_integer('batch_size', 20,"""mini-batch size""")
flags.DEFINE_float('max_grad_norm', 10,"""gradient clipping norm""")
flags.DEFINE_float('fisher_noise', 0.01,"""Fisher noise level""")
flags.DEFINE_integer('dataset_size', 26400, "Size of the datatset")
flags.DEFINE_integer('max_epoch', 112, "first decay of the learning rate")
flags.DEFINE_integer('max_max_epoch', 440, "total number of epochs trained")
flags.DEFINE_float('power', 1.0, """FVPS power""")
flags.DEFINE_float('learning_rate', 1.0, "learning rate chosen")
FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    # self._is_training = is_training
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
    self.max_grad_norm = max_grad_norm = config.max_grad_norm
    self.learning_rate = learning_rate = config.learning_rate
    self.max_epoch = max_epoch = config.max_epoch
    self.max_max_epoch = max_max_epoch = config.max_max_epoch

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_
    self._is_training = is_training
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    if config.hidden_size < 250:
      lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    else:
      lstm_cell = rnn_cell.BasicLSTMCell_block(size, forget_bias=0.0, state_is_tuple=True)

    if is_training and config.keep_prob < 1:
      lstm_cell = rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      #inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      input_shape = input_.input_data.get_shape()
      input_flatten = tf.reshape(input_.input_data, [-1, 1])
      #batch_size_flatten = tf.size(input_flatten)[0]
      batch_size_flatten = int(input_shape[0]*input_shape[1])
      indices = tf.expand_dims(tf.range(0, batch_size_flatten, 1), 1)
      concated = tf.concat(axis=1, values=[indices, input_flatten])
      onehot_input = tf.sparse_to_dense(
          concated, tf.stack([batch_size_flatten, vocab_size]), 1.0, 0.0)
      inputs = tf.reshape(tf.matmul(onehot_input, embedding), [int(input_shape[0]), int(input_shape[1]), size])

      print(input_.input_data)
      print(inputs)
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = [tf.squeeze(input_step, [1])
    #           for input_step in tf.split(1, num_steps, inputs)]
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)

    ### cost function
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    ###
    ### sample cost function
    noise_shape = tf.concat(axis=0, values=[tf.shape(logits), [1,]])
    gumbel_noise = tf.transpose(-tf.log(-tf.log(tf.random_uniform(noise_shape)+1e-8)), [0,2,1])
    sampled_labels = tf.to_int32( tf.reshape(tf.argmax(tf.expand_dims(logits, 1) + gumbel_noise, 2), [-1]) )
    sampled_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [sampled_labels],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._sampled_cost = sampled_cost = tf.reduce_sum(sampled_loss) / batch_size
    ###
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()

    #################
    #################
    self._QR = None
    useSGD = False


    gs = tf.gradients(cost, tvars, name='gradientsSampled') ## HACK: have to use this name!!!)
    grads, _ = tf.clip_by_global_norm(gs, config.max_grad_norm)

    noise = [tf.random_normal(tf.shape(v)) for v in tvars]
    ############## estimate curvature (Fisher)
    cov = cov_estimator.Estimator(approx_option='diag') ##kfac option current broken
    cov.compute_stats(cost, tvars, gs)
    fvps = cov.FisherVectorProduct(zip(tvars,noise), power = FLAGS.power, fisher_noise = FLAGS.fisher_noise)
    # fvps[0] = tf.Print(fvps[0], fvps)
    ##############
    grads_FisherNoise = [g + FishNoise for g, FishNoise in zip(grads, fvps)]

    if FLAGS.useSGD:
      self.optimizer = tf.train.GradientDescentOptimizer(self._lr)
      self._train_op = self.optimizer.apply_gradients(
          zip(grads_with_noise, tvars),
          global_step=tf.contrib.framework.get_or_create_global_step())

    elif FLAGS.useAdam:
      self.optimizer = tf.train.AdamOptimizer(self._lr*0.001)
      self._train_op = self.optimizer.apply_gradients(
          zip(grads_with_noise, tvars),
          global_step=tf.contrib.framework.get_or_create_global_step())

    elif FLAGS.useSGD_diagF:
      self.optimizer = tf.train.GradientDescentOptimizer(self._lr)
      self._train_op = self.optimizer.apply_gradients(
          zip(grads_FisherNoise, tvars),
          global_step=tf.contrib.framework.get_or_create_global_step())

    elif FLAGS.useAdam_diagF:
      self.optimizer = tf.train.AdamOptimizer(self._lr*0.001)
      self._train_op = self.optimizer.apply_gradients(
          zip(grads_FisherNoise, tvars),
          global_step=tf.contrib.framework.get_or_create_global_step())


    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  # def export_ops(self, name):
    # """Exports ops to collections."""
    # self._name = name
    # ops = {util.with_prefix(self._name, "cost"): self._cost}
    # if self._is_training:
      # ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      # if self._rnn_params:
        # ops.update(rnn_params=self._rnn_params)
    # for name, op in ops.items():
      # tf.add_to_collection(name, op)
    # self._initial_state_name = util.with_prefix(self._name, "initial")
    # self._final_state_name = util.with_prefix(self._name, "final")
    # util.export_state_tuples(self._initial_state, self._initial_state_name)
    # util.export_state_tuples(self._final_state, self._final_state_name)

  # def import_ops(self):
    # """Imports ops from collections."""
    # if self._is_training:
      # self._train_op = tf.get_collection_ref("train_op")[0]
      # self._lr = tf.get_collection_ref("lr")[0]
      # self._new_lr = tf.get_collection_ref("new_lr")[0]
      # self._lr_update = tf.get_collection_ref("lr_update")[0]
      # rnn_params = tf.get_collection_ref("rnn_params")
  # #     if self._cell and rnn_params:
# #         params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            # # self._cell,
            # # self._cell.params_to_canonical,
            # # self._cell.canonical_to_params,
            # # rnn_params,
       # #      base_variable_scope="Model/RNN")
 # #        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    # self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    # num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    # self._initial_state = util.import_state_tuples(
        # self._initial_state, self._initial_state_name, num_replicas)
    # self._final_state = util.import_state_tuples(
        # self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = FLAGS.max_grad_norm
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = FLAGS.max_epoch
  max_max_epoch = FLAGS.max_max_epoch
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = FLAGS.batch_size
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = FLAGS.max_grad_norm
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = FLAGS.max_epoch
  #max_max_epoch = 59
  max_max_epoch = FLAGS.max_max_epoch
  keep_prob = 0.5
  lr_decay = 0.8
  #lr_decay = 0.92
  batch_size = FLAGS.batch_size
  vocab_size = 10000

#class MediumConfig(object):
#  """Medium config."""
#  init_scale = 0.05
#  learning_rate = 1.0
#  max_grad_norm = 5
#  num_layers = 2
#  num_steps = 35
#  hidden_size = 650
#  max_epoch = 6
#  max_max_epoch = 39
#  keep_prob = 0.5
#  lr_decay = 0.8
#  batch_size = 20
#  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = FLAGS.learning_rate
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = FLAGS.max_epoch
  max_max_epoch = FLAGS.max_max_epoch
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = FLAGS.batch_size
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  if eval_op is not None:
    # if not FLAGS.useSGD_diagF:
    if not FLAGS.useAdam_diagF:
    # if not (FLAGS.useSGD and FLAGS.useAdam and FLAGS.useSGD_diagF and
            # FLAGS.useAdam_diagF):
    # if not FLAGS.useSGD and not FLAGS.useAdam:
      fetches["vF"] = model.optimizer.vFv

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      if eval_op is not None:
        if not FLAGS.useAdam_diagF:
        # if not FLAGS.useSGD_diagF:
#         if not FLAGS.useSGD and not FLAGS.useAdam:
          print("vFv: %f"%vals["vF"])
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()

  # print the script flags
  print('##################')
  for key in FLAGS.__dict__['__flags']:
    print(key, ': ', FLAGS.__dict__['__flags'][key])
  print('##################')
  print({attr:config.__getattribute__(attr) for attr in dir(config)
              if not attr.startswith('__')})


  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      if m._QR:
            print("starting KFAC queue")
            coord = tf.train.Coordinator()
            enqueue_threads = m._QR.create_threads(session, coord=coord, start=True)

      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        iterations = (FLAGS.dataset_size // FLAGS.batch_size) * (i+1)
        print("Iterations: %d" % iterations)

        with open('/home/kevin/PTBIterationsResults/PTBFisher' + FLAGS.saved_path + '.csv' ,'a+') as f:
            f.write("%d,%.3f,%.3f,%d \n" % (i + 1, train_perplexity, valid_perplexity, iterations))
      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()


