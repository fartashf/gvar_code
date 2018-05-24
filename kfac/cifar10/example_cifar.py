import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
import time, os
#import cifar10_input
import rbg_cifar
import kfac
import resnet_model

sys.path.append('../../')
from scheduler import DMomScheduler
from log_utils import LogCollector, TBXWrapper
tb_logger = TBXWrapper()

# CIFAR_DATA_DIR = '%s/datasets/cifar-10-batches-bin'%os.getenv("HOME")
CIFAR_DATA_DIR = '../../data/cifar10/cifar-10-batches-py/'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('name', 'kfc_debug',"""experiment name""")
tf.app.flags.DEFINE_boolean('debug', False,"""turn on debug flag""")
######### OPTIMIZER PARAM ##########
tf.app.flags.DEFINE_float('lr', 0.01,"""Learning rate""")
tf.app.flags.DEFINE_float('mom', 0.9,"""Momentum""")
tf.app.flags.DEFINE_float('paramMovingAverage', 0.98,"""Parameter moving average momentum""")
tf.app.flags.DEFINE_float('weightDecayConv', 0.004,"""weight decay conv layer""")
tf.app.flags.DEFINE_float('weightDecayFC', 1.,"""weight decay for the linear layer""")
tf.app.flags.DEFINE_float('eps', 1e-3,"""damping factor""")
tf.app.flags.DEFINE_float('statDecay', 0.95,"""decay running average of stats factor""")
tf.app.flags.DEFINE_boolean('fullStatsInit', True,"""Initialize stats by accumulating a super batch""")
tf.app.flags.DEFINE_boolean('useAsync', False,"""Use asynchronous updates""")
tf.app.flags.DEFINE_boolean('useSGD', False,"""Use SGD optimizer instead""")
tf.app.flags.DEFINE_boolean('useAdam', False,"""Use Adam optimizer instead""")
tf.app.flags.DEFINE_integer('colditer', 100,"""number of cold start iteration using SGD""")
###################
tf.app.flags.DEFINE_integer('batchSize', 512,"""mini-batch size""")
tf.app.flags.DEFINE_integer('test_batchsize', 1000 ,"""test mini-batch   size""")
tf.app.flags.DEFINE_integer('randomseed', 1234 ,"""random seed""")
tf.app.flags.DEFINE_integer('maxiter', 100000 ,"""maximum updates""")
tf.app.flags.DEFINE_integer('validiter', 0 ,"""validation updates""")
################# Dmom Scheduler
tf.app.flags.DEFINE_boolean('useDmom', False,"""Use Dmom Scheduler""")
tf.app.flags.DEFINE_float('dmom', 0.9 ,"""Data momentum coefficient""")
tf.app.flags.DEFINE_string('alpha_norm', 'sum' ,"""sum|none normalization""")
tf.app.flags.DEFINE_string('sampler_params', '50,90,2,200' ,"""Scheduler params""")
tf.app.flags.DEFINE_string('sampler_w2c', 'linrank' ,"""Weight to Counts""")
tf.app.flags.DEFINE_integer('sampler_maxw', 1 ,"""Maximum weight""")
tf.app.flags.DEFINE_integer('sampler_start_epoch', 1 ,"""start epoch""")
tf.app.flags.DEFINE_string('logger_name', 'runs/X' ,"""""")
tf.app.flags.DEFINE_boolean('log_nex', False ,"""""")
tf.app.flags.DEFINE_string('log_keys', '' ,"""""")
tf.app.flags.DEFINE_boolean('use_resnet', False ,"""""")

tf.set_random_seed(FLAGS.randomseed)

BATCHSIZE = FLAGS.batchSize
FAC_UPDATE = FLAGS.colditer
kfac.KFAC_DEBUG = FLAGS.debug
# print the script flags
print '##################'
for key, value in FLAGS.flag_values_dict().iteritems():
  print key, ': ', value
print '##################'

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]


opt = DictWrapper(FLAGS.flag_values_dict())
opt.batch_size = opt.batchSize
tb_logger.configure(opt.logger_name, flush_secs=5, opt=opt)

NUM_CIFAR_TRAIN = 50000
USE_BLOCKDIAG_BIAS = False

validUpdate = FLAGS.validiter
if FLAGS.validiter <= 0:
    validUpdate = int(float(NUM_CIFAR_TRAIN)/BATCHSIZE)

import math
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
IMAGE_SIZE = 32
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3
NUMHIDDEN = 32

NUM_TEST_EXAMPLES = 10000
TEST_BATCHSIZE = FLAGS.test_batchsize


class BatchIterator:
  def __init__(self, data, subset):
    self.data = data
    self.subset = subset
    self.start = 0

  def get_X_y(self):
    if self.subset == 'train':
      return self.data.X_train_shuf, self.data.y_train_shuf, self.data.index_shuf
    elif self.subset == 'train_test':
      return self.data.X_train, self.data.y_train, None
    elif self.subset == 'test':
      return self.data.X_test, self.data.y_test, None
    else:
      raise RuntimeError('Unknown subset: {}'.format(self.subset))

  def next_batch(self, mbsize):
    end = self.start + mbsize
    X, y, idx = self.get_X_y()

    if end > X.shape[0]:
      if self.subset == 'train':
        self.data.shuffle()
      X, y, idx = self.get_X_y()
      self.start = 0
      end = self.start + mbsize

    start = self.start
    self.start += mbsize

    I = None
    if idx is not None:
        I = idx[start:end]

    if self.data.scheduler and self.subset == 'train':
        self.data.scheduler.niters += 1
    return X[start:end, ...].reshape((mbsize, 32, 32, 3), order='F'), y[start:end, ...], I

  def update_alpha(self, idx, alpha):
      self.data.update_alpha(idx, alpha)

def fill_feed_dict(data_set, images_pl, labels_pl, batchsize):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed, idx = data_set.next_batch(batchsize)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict, idx




def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_mean(tf.cast(correct, tf.float32))


def NLLLoss(logits, labelsPL, scope):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  # Convert from sparse integer labels in the range [0, NUM_CLASSES)
  # to 1-hot dense float vectors (that is we will have batch_size vectors,
  # each with NUM_CLASSES values, all of which are 0.0 except there will
  # be a 1.0 in the entry corresponding to the label).
  batch_size = tf.size(labelsPL)
  labels = tf.expand_dims(labelsPL, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(axis=1, values=[indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits, labels=onehot_labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  alpha = cross_entropy

  # get weight decay losses
  weight_losses = tf.get_collection('losses', scope)
  if len(weight_losses) > 0:
      weight_losses = tf.add_n(weight_losses)

  return loss + weight_losses, alpha


def NLLLoss_sampled(logits, numSamples = 1):
  logits_expanded = tf.tile(logits, [numSamples, 1])
  noise_shape = tf.concat(axis=0, values=[tf.shape(logits), [numSamples,]])
  gumbel_noise = tf.transpose(-tf.log(-tf.log(tf.random_uniform(noise_shape)+1e-8)), [0,2,1])
  sampled_labels = tf.to_int32( tf.reshape(tf.argmax(tf.expand_dims(logits, 1) + gumbel_noise, 2), [-1]) )
  batch_size = tf.size(sampled_labels)
  labels = tf.expand_dims(sampled_labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(axis=1, values=[indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits_expanded, labels=onehot_labels, name='xentropy_sample')
  loss_sampled = tf.reduce_mean(cross_entropy, name='xentropy_mean_sample')
  return loss_sampled


def Linear(inputX, nIn, nOut, name, reuse_avg=None, weightLossDict=None):
  with tf.variable_scope(name):
    #stddev = 1. / math.sqrt(float(nIn))
    stddev = 0.01
    weights = tf.get_variable('weights', [nIn, nOut],
                              initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable('biases', [nOut], initializer=tf.constant_initializer(0.))
    if reuse_avg:
        weights = reuse_avg.average(weights)
        biases = reuse_avg.average(biases)
    if FLAGS.weightDecayFC > 0.:
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), FLAGS.weightDecayFC, name='weight_decay_loss')
        if weightLossDict is not None:
            weightLossDict[weights] = FLAGS.weightDecayFC
            weightLossDict[biases] = 0.
        tf.add_to_collection('losses', weight_decay)
    return tf.nn.bias_add(tf.matmul(inputX, weights), biases)

def Conv(inputX, nIn, nOut, ks, name, reuse_avg=None, weightLossDict=None):
  with tf.variable_scope(name):
    #stddev = 1. / math.sqrt(float(nIn))
    stddev = 0.01
    weights = tf.get_variable('weights', [ks, ks, nIn, nOut],
                              initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable('biases', [nOut], initializer=tf.constant_initializer(0.))
    if reuse_avg:
        weights = reuse_avg.average(weights)
        biases = reuse_avg.average(biases)
    if FLAGS.weightDecayConv > 0.:
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), FLAGS.weightDecayConv, name='weight_decay_loss')
        if weightLossDict is not None:
            weightLossDict[weights] = FLAGS.weightDecayConv
            weightLossDict[biases] = 0.
        tf.add_to_collection('losses', weight_decay)
    return tf.nn.bias_add(tf.nn.conv2d(inputX, weights, [1, 1, 1, 1], padding='SAME'), biases)

def MaxPool(inputX, ks, stride):
  return tf.nn.max_pool(inputX, [1, ks, ks, 1], [1, stride, stride, 1], padding='SAME')

def buildModel_basic(inputX, weightLossDict=None, reuse_avg=None, *args, **kwargs):
  # TODO: response normalization (tf.nn.local_response_normalization)
  #batch_size = inputX.get_shape()[0]

  # TEMPORARY
  if reuse_avg:
    batch_size = TEST_BATCHSIZE
  else:
    batch_size = BATCHSIZE

  conv1 = Conv(inputX, 3, 32, 5, 'conv1', reuse_avg=reuse_avg, weightLossDict=weightLossDict)
  pool1 = tf.nn.relu(MaxPool(conv1, 3, 2))
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  conv2 = (Conv(pool1, 32, 32, 5, 'conv2', reuse_avg=reuse_avg, weightLossDict=weightLossDict))
  pool2 = tf.nn.relu(MaxPool(conv2, 3, 2))
  norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

  conv3 = Conv(norm2, 32, 64, 5, 'conv3', reuse_avg=reuse_avg, weightLossDict=weightLossDict)
  pool3 = tf.nn.relu(MaxPool(conv3, 3, 2))
  pool3_rsh = tf.reshape(pool3, [-1, 4*4*64])
  out = Linear(pool3_rsh, 4*4*64, 10, 'out', reuse_avg=reuse_avg, weightLossDict=weightLossDict)

  return out


def buildModel(inputX, *args, **kwargs):
    return buildModel_basic(inputX, *args, **kwargs)

def run_kfac():
  data = rbg_cifar.load_data(rbg_cifar.DataParams(unit_variance=False))
  if FLAGS.useDmom:
      # opt = {'dmom': FLAGS.dmom, 'alpha_norm': FLAGS.alpha_norm,
      #        'sampler_params': FLAGS.sampler_params,
      #        'sampler_w2c': FLAGS.sampler_w2c,
      #        'sampler_maxw': FLAGS.sampler_maxw, 'batch_size': BATCHSIZE,
      #        'sampler_start_epoch': FLAGS.sampler_start_epoch,
      #        'maxiter': FLAGS.maxiter}
      # opt = DictWrapper(opt)
      data.scheduler = DMomScheduler(data.X_train.shape[0], opt)

  #train_images, train_labels = cifar10_input.inputs(
  #  eval_data=False, data_dir=CIFAR_DATA_DIR, batch_size=BATCHSIZE, image_size=32)
  #test_images, test_labels = cifar10_input.inputs(
  #  eval_data=True, data_dir=CIFAR_DATA_DIR, batch_size=TEST_BATCHSIZE, image_size=32)
  # Get data queues
  data_qrs = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)


  assert NUM_TEST_EXAMPLES % TEST_BATCHSIZE == 0, 'TEST_BATCHSIZE must divide NUM_TEST_EXAMPLES'
  num_test_batches = NUM_TEST_EXAMPLES // TEST_BATCHSIZE

  print 'AAA'



  with tf.device('/gpu'):

     train_images = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
     train_labels = tf.placeholder(tf.int32, shape=(None))
     # train_images = tf.placeholder(tf.float32, shape=(BATCHSIZE, 32, 32, 3))
     # train_labels = tf.placeholder(tf.int32, shape=(BATCHSIZE))

     test_images = train_images
     test_labels = train_labels
     # test_images = tf.placeholder(tf.float32, shape=(TEST_BATCHSIZE, 32, 32, 3))
     # test_labels = tf.placeholder(tf.int32, shape=(TEST_BATCHSIZE))

     weightLossDict = {}

     if opt.use_resnet:
        resnet32 = resnet_model.Cifar10Model(32)

     with tf.name_scope('train') as scope:
         if opt.use_resnet:
             model_out = resnet32(train_images, training=True)
         else:
            model_out = buildModel(train_images, weightLossDict = weightLossDict, mode='train')
         total_loss, alpha = NLLLoss(model_out, train_labels, scope)
         acc = evaluation(model_out, train_labels)
         #tf.get_variable_scope().reuse_variables()

         # Compute gradients.
         queue_runner = None
         #USE SGD
         if FLAGS.useSGD:
            updates = tf.train.MomentumOptimizer(learning_rate = FLAGS.lr * (1. - FLAGS.mom), momentum = FLAGS.mom).minimize(total_loss)
         elif FLAGS.useAdam:
            updates = tf.train.AdamOptimizer().minimize(total_loss)
         #USE K-FAC
         else:
            loss_sampled = NLLLoss_sampled(model_out)
            updates, queue_runner = kfac.KfacOptimizer(learning_rate=FLAGS.lr, momentum=FLAGS.mom, epsilon=FLAGS.eps, stats_decay=FLAGS.statDecay, blockdiag_bias=USE_BLOCKDIAG_BIAS, async=FLAGS.useAsync, full_stats_init=FLAGS.fullStatsInit, cold_iter=FLAGS.colditer, weight_decay_dict=weightLossDict, approxT2 = False, factored_damping=True).minimize(total_loss, loss_sampled)

     # pk parameter averaging one epoch
     averager = tf.train.ExponentialMovingAverage(decay=FLAGS.paramMovingAverage)
     update_averages = averager.apply(tf.trainable_variables())

     learning_ops = [updates,]
     with tf.control_dependencies(learning_ops):
         train_op = tf.group(update_averages)

     with tf.name_scope('test') as scope:
         tf.get_variable_scope().reuse_variables()
         if opt.use_resnet:
             model_out = resnet32(train_images, training=False)
         else:
            model_out_test = buildModel(test_images, reuse_avg=averager, mode='test')
         total_loss_test, _ = NLLLoss(model_out_test, test_labels, scope)
         acc_test = evaluation(model_out_test, test_labels)



  print 'BBB'


  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        intra_op_parallelism_threads=8,
                                        )) as session:

     saver = tf.train.Saver(var_list=tf.trainable_variables()) # saves variables learned during training
     tf.global_variables_initializer().run()
     #saver.restore(session, 'cifar10modelcold.ckpt') # to restore from model, uncomment this line
     print 'CCC'
     print 'validUpdate =', validUpdate

     coord = tf.train.Coordinator()
     #threads = tf.train.start_queue_runners(sess=session, coord=coord)
     enqueue_threads = []
     for qr in data_qrs:
        print("starting data queue")
        t = qr.create_threads(session, coord=coord, start=True)
        enqueue_threads.extend(t)
     if queue_runner is not None:
        print("starting kfac queue")
        enqueue_threads.extend(queue_runner.create_threads(session, coord=coord, start=True))

     train_iter = BatchIterator(data, 'train')
     train_test_iter = BatchIterator(data, 'train_test')
     test_iter = BatchIterator(data, 'test')

     train_acc_tol = 0
     train_loss_tol = 0
     train_time_tol = 0
     start_time = time.time()
     for i in range(FLAGS.maxiter):
       feed_dict, idx = fill_feed_dict(train_iter,
                            train_images,
                            train_labels,
                            BATCHSIZE)
       _, Loss, Acc, alpha_val = session.run([train_op, total_loss, acc, alpha], feed_dict=feed_dict)
       train_iter.update_alpha(idx, alpha_val)

       #print 'DDD', i
       train_acc_tol += Acc
       train_loss_tol += Loss
       if (i+1)%validUpdate == 0:
           test_acc_tol = 0
           test_loss_tol = 0
           train_acc_tol_avg = 0
           train_loss_tol_avg = 0
           duration = time.time() - start_time
           for j in range(int(NUM_CIFAR_TRAIN/TEST_BATCHSIZE)):
             feed_dict, _ = fill_feed_dict(train_test_iter,
                              train_images,
                              train_labels,
                              TEST_BATCHSIZE)

             train_loss_avg, train_acc_avg = session.run([total_loss_test, acc_test], feed_dict=feed_dict)

             train_loss_tol_avg += train_loss_avg
             train_acc_tol_avg += train_acc_avg

           for j in range(int(num_test_batches)):
             feed_dict, _ = fill_feed_dict(test_iter,
                              train_images,
                              train_labels,
                              TEST_BATCHSIZE)

             test_loss, test_acc = session.run([total_loss_test, acc_test], feed_dict=feed_dict)

             test_loss_tol += test_loss
             test_acc_tol += test_acc

           train_acc_tol /= float(validUpdate)
           train_acc_tol_avg /= float(int(NUM_CIFAR_TRAIN/TEST_BATCHSIZE))
           test_acc_tol /= float(num_test_batches)
           train_loss_tol /= float(validUpdate)
           train_loss_tol_avg /= float(int(NUM_CIFAR_TRAIN/TEST_BATCHSIZE))
           test_loss_tol /= float(num_test_batches)
           print 'update : ', int(i), time.strftime("%D %H:%M:%S")
           print 'train loss: ',train_loss_tol,' train acc: ',train_acc_tol
           print 'train loss avg: ',train_loss_tol_avg,' train acc avg: ',train_acc_tol_avg
           print 'test loss: ', test_loss_tol, ' test acc: ', test_acc_tol
           print validUpdate*BATCHSIZE/duration, ' images per sec.'
           train_time_tol += duration
           start_time = time.time()
           train_acc_tol = 0
           train_loss_tol = 0

           tb_logger.log_value('Tloss', train_loss_tol_avg, step=i)
           tb_logger.log_value('Tacc', train_acc_tol_avg*100, step=i)
           tb_logger.log_value('Vloss', test_loss_tol, step=i)
           tb_logger.log_value('Vacc', test_acc_tol*100, step=i)
           logger = LogCollector(opt)
           if opt.useDmom:
               logger.update('count_h',
                             train_iter.data.scheduler.count, 1, hist=True)
               logger.update('visits_h',
                             train_iter.data.scheduler.visits, 1, hist=True)
               print(str(logger))
               logger.tb_log(tb_logger, step=i)
           tb_logger.save_log()

           #print("Model saved in file: %s" % saver.save(session, 'cifar10modelcold.ckpt'))

     coord.request_stop()
     coord.join(enqueue_threads)
     session.close()




run_kfac()


