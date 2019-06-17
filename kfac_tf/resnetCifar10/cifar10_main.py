# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import numpy as np
import resnet_model
sys.path.append('../')
import cov_estimator
import kfac
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adamnoisy import AdamNoisyOptimizer

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The base directory where the model will be stored.')

parser.add_argument('--resnet_size', type=int, default=32,
                    help='The size of the ResNet model to use.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument(
    '--data_format', type=str, default='channels_last',
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

parser.add_argument('--logdir', type=str, default='./logs',
                    help='Base directory where the results are stored.')

parser.add_argument('--useNoiseType', type=str, default='None',
                    help="Add gradient noise types: 'Gaussian', 'DiagF', 'DiagF2' squared Diag Fisher.")

######### OPTIMIZER PARAM ##########
parser.add_argument('--optimizer', type=str, default='SGDMom',
                    help="Name of optimizer: 'SGDMom', 'Gauss'")

parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate.')

parser.add_argument('--mom', type=float, default=0.9,
                    help='Momentum.')

parser.add_argument('--eps', type=float, default=1e-8,
                    help='Damping Factor for Adam.')

parser.add_argument('--noiseLambda', type=float, default=0.0,
                    help='Lambda for fisher noise.')

parser.add_argument('--refBatchSize', type=int, default=25,
                    help='Size of the Reference Batch Size when doing DiffNoise noise Type. Default: 25')


_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

def plot_results(path, title_train, title_test, updates, train_acc, test_acc, mode, opt, lr, lb, batch_size, isAcc, train_min_y=0.0, train_max_y=1.0, val_min_y=0.0, val_max_y=1.0):
    # Plot Training Error
    ylabel = "Training Error"
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.title(title_train)
    plt.ylabel(ylabel)
    plt.xlabel('Parameter Updates')

    # Plot Validation Error
    plt.subplot(1, 2, 2)
    ylabel = "Test Error"
    plt.title(title_test)
    plt.ylabel(ylabel)
    plt.xlabel('Parameter Updates')

    plt.subplot(1, 2, 1)
    if isAcc: # Invert 1 - value
        plt.plot(updates, 1-train_acc, label="{} {}".format(batch_size, mode))
    else:
        plt.plot(updates, train_acc, label="{} {}".format(batch_size, mode))

    plt.ylim([train_min_y,train_max_y])
    plt.legend()

    plt.subplot(1, 2, 2)   
    if isAcc:         
        plt.plot(updates, 1-test_acc, label="{} {}".format(batch_size, mode))
    else:
        plt.plot(updates, test_acc, label="{} {}".format(batch_size, mode))

    plt.ylim([val_min_y,val_max_y])
    plt.legend()

    # Save figure
    plt.savefig(path)
    plt.close()
    return

def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record):
  """Parse CIFAR-10 image and label from a raw record."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  label_bytes = 1
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes

  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = record_dataset(get_filenames(is_training, data_dir))

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance. Because CIFAR-10
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training), label))

  dataset = dataset.prefetch(2 * batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Batch results by up to batch_size, and then fetch the tuple from the
  # iterator.
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  tf.summary.image('images', features, max_outputs=6)

  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES, params['data_format'])

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  # loss = cross_entropy  # TODO: Debugging without weight decay when adding noise. See if there's still Add_N Error 

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Get all the trainable variables and their gradients
    var = tf.trainable_variables()    

    if FLAGS.useNoiseType == "DiagF":
        # Get the gradient wrt to xentropy loss (but not the weight decay term)
        cross_entropy_mean = tf.identity(cross_entropy, name='xentropy_mean')
        gs_xentropy = tf.gradients(cross_entropy_mean, var, name='gradientsSampled') ## HACK: have to use this name!!!
        noise = [tf.random_normal(tf.shape(v)) for v in var]
        noise_sq = [tf.square(n) for n in noise]
        ############## estimate curvature (Fisher)
        cov = cov_estimator.Estimator(approx_option='diag') 
        cov.compute_stats(cross_entropy_mean, var, gs_xentropy)
        fvps = cov.FisherVectorProduct(zip(var,noise), power = 0.5) * FLAGS.noiseLambda
        ##############
        
        gs = tf.gradients(loss, var)
        #### Add the Fisher noise into the gradients if not AdamNoisy Optimizer
        if not FLAGS.optimizer == "AdamNoisy":
            gs = [g + f_noise for g, f_noise in zip(gs, fvps)]
        ####
    elif FLAGS.useNoiseType == "Gauss":
        gs = tf.gradients(loss, var)
        noise = [FLAGS.noiseLambda * tf.random_normal(tf.shape(v)) for v in var]
        #### Add the Fisher noise into the gradients if not AdamNoisy Optimizer
        if not FLAGS.optimizer == "AdamNoisy":
            gs = [g + n for g, n in zip(gs, noise)]
    elif FLAGS.useNoiseType == "DiffNoise":
        # Compute the loss for minibatch
        gs = tf.gradients(loss, var)
        cross_entropy_ref1 = tf.losses.softmax_cross_entropy(logits=logits[:FLAGS.refBatchSize], onehot_labels=labels[:FLAGS.refBatchSize])
        cross_entropy_ref2 = tf.losses.softmax_cross_entropy(logits=logits[-FLAGS.refBatchSize:], onehot_labels=labels[-FLAGS.refBatchSize:])
        gs_1 = tf.gradients(cross_entropy_ref1, var)
        gs_2 = tf.gradients(cross_entropy_ref2, var)
        noise = [FLAGS.noiseLambda*tf.stop_gradient(g1 - g2) for g1, g2 in zip(gs_1, gs_2)]

        if not FLAGS.optimizer == "AdamNoisy":
            gs = [g + n for g, n in zip(gs, noise)]

    elif FLAGS.useNoiseType == "None":
        pass
    else:
        print("Error: Unrecognized --optimizer option. Choose from: 'SGDMom', 'Adam'")
        return

    # Decide on the optimizer    
    global_step = tf.train.get_or_create_global_step()
    if FLAGS.optimizer == "SGDMom":
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 128, the learning rate should be 0.1.
        # initial_learning_rate = 0.1 * params['batch_size'] / 128
        initial_learning_rate = FLAGS.lr
        batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in [100 * params['batch_size'] / 128, 150* params['batch_size'] / 128, 200 *params['batch_size'] / 128]] 
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if FLAGS.useNoiseType == "None":
                train_op = optimizer.minimize(loss, global_step=global_step)
            else:
                train_op = optimizer.apply_gradients(zip(gs, var), global_step=global_step)
    
    elif FLAGS.optimizer == "Adam":
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if FLAGS.useNoiseType == "None":
                train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, epsilon=FLAGS.eps).minimize(loss, global_step=global_step)
            else:
                train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, epsilon=FLAGS.eps).apply_gradients(zip(gs, var), global_step=global_step)
            
    elif FLAGS.optimizer == "AdamNoisy":
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if FLAGS.useNoiseType == 'DiagF':
                train_op = AdamNoisyOptimizer(learning_rate=FLAGS.lr, epsilon=FLAGS.eps).apply_gradients(zip(gs, var), fvps, global_step=global_step)
            elif FLAGS.useNoiseType == 'Gauss' or FLAGS.useNoiseType == 'DiffNoise':
                train_op = AdamNoisyOptimizer(learning_rate=FLAGS.lr, epsilon=FLAGS.eps).apply_gradients(zip(gs, var), noise, global_step=global_step)
            elif FLAGS.useNoiseType == "None":
                print("Invalid flag combination: AdamNoisy needs to use noiseType that is not None")
                return 
    
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  
  # Create the folder for the results
  mode = FLAGS.useNoiseType
  optimizer_name = FLAGS.optimizer
  if FLAGS.optimizer == "Adam":
      learning_rate_str = "LR{}eps{}".format(FLAGS.lr, FLAGS.eps)
  else:
      learning_rate_str = "LR{}".format(FLAGS.lr)
  bs_str = "bs{}".format(FLAGS.batch_size)
  noise_lambda_str = "lambda0" if mode is "None" else "lambda{}".format(FLAGS.noiseLambda)
  path = os.path.join(FLAGS.model_dir, mode, optimizer_name, learning_rate_str, noise_lambda_str, bs_str)
  if not os.path.exists(path):
    os.makedirs(path)
    print("Created logdir path {}".format(path))

  with open(os.path.join(path, 'experiment_flags.txt'), 'w') as f:
    flags_dict = vars(FLAGS)
    hparam_string = ''
    for key in flags_dict:
        hparam_string += "{}={}\n".format(key, flags_dict[key])
    f.write(hparam_string)
  
  model_dir = path

  cifar_classifier = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir=model_dir, config=run_config,
      params={
          'resnet_size': FLAGS.resnet_size,
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })

  updates_list = []
  train_loss_list = []
  test_loss_list = []
  train_acc_list = []
  test_acc_list = []

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    if FLAGS.optimizer == 'SGDMom':
      tensors_to_log = {
          'learning_rate': 'learning_rate',
          'cross_entropy': 'cross_entropy',
          'train_accuracy': 'train_accuracy'
      }
    elif FLAGS.optimizer == 'Adam' or FLAGS.optimizer == 'AdamNoisy':
      tensors_to_log = {
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
      }
    else:
      print("Unrecognized optimizer. Choose from 'Adam', 'AdamNoisy', or 'SGDMom' for now.")

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    cifar_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_results = cifar_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print("Printing eval results:")
    print(eval_results)

    # Get the training data model results
    train_eval_results = cifar_classifier.evaluate(
        input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, num_epochs=1))
    print("Printing train eval results:")
    print(train_eval_results)
    
    update_iteration = _ * FLAGS.epochs_per_eval * float(_NUM_IMAGES['train']) / FLAGS.batch_size
    print("Update iteration {}: Test Acc: {}, Test Loss: {}, Train Acc: {}, Train Loss: {}".format(update_iteration, eval_results['accuracy'], eval_results['loss'], train_eval_results['accuracy'], train_eval_results['loss']))
    updates_list.append(update_iteration)
    test_acc_list.append(eval_results['accuracy'])
    test_loss_list.append(eval_results['loss'])
    train_acc_list.append(train_eval_results['accuracy'])
    train_loss_list.append(train_eval_results['loss'])
    with open(os.path.join(path,'results.csv'), 'w') as fp:
        fp.write(','.join(['updates','train_loss', 'train_acc', 'test_loss', 'test_acc']) + '\n')
        np.savetxt(fp, np.array([updates_list,train_loss_list, train_acc_list, test_loss_list, test_acc_list]).transpose(), '%s', ',')

    # Plot the error curves so far:
    title_train = "Training Error over Updates {} {} {}".format(optimizer_name, learning_rate_str, noise_lambda_str) 
    title_test = "Testing Error over Updates {} {} {}".format(optimizer_name, learning_rate_str, noise_lambda_str) 
    path_fig = os.path.join(path, "Error_Plot.png")
    plot_results(path_fig, title_train, title_test, np.array(updates_list), np.array(train_acc_list), np.array(test_acc_list), mode, optimizer_name, learning_rate_str, noise_lambda_str, bs_str, True, train_min_y=0.0, train_max_y=1.0, val_min_y=0.0, val_max_y=1.0)

    title_train = "Training Loss over Updates {} {} {}".format(optimizer_name, learning_rate_str, noise_lambda_str) 
    title_test = "Testing Loss over Updates {} {} {}".format(optimizer_name, learning_rate_str, noise_lambda_str) 
    path_fig = os.path.join(path, "Loss_Plot.png")
    plot_results(path_fig, title_train, title_test, np.array(updates_list), np.array(train_loss_list), np.array(test_loss_list), mode, optimizer_name, learning_rate_str, noise_lambda_str, bs_str, False, train_min_y=0.0, train_max_y=1.0, val_min_y=0.0, val_max_y=1.0)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
