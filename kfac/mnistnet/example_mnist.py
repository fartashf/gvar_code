import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import time
import mnist_data
#import kfac
import async_wrapper as kfac

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('name', 'kfc_debug',"""experiment name""")
tf.app.flags.DEFINE_boolean('debug', False,"""turn on debug flag""")
tf.app.flags.DEFINE_boolean('flip', False,"""flip the mnist pixels""")
tf.app.flags.DEFINE_boolean('useFCModel', False,"""Use fully connected model""")
######### OPTIMIZER PARAM ##########
tf.app.flags.DEFINE_float('lr', 0.001,"""Learning rate""")
tf.app.flags.DEFINE_float('mom', 0.9,"""Momentum""")
tf.app.flags.DEFINE_float('eps', 1e-2,"""damping factor""")
tf.app.flags.DEFINE_float('statDecay', 0.95,"""decay running average of stats factor""")
tf.app.flags.DEFINE_boolean('useAsync', False,"""Use asynchronous updates""")
tf.app.flags.DEFINE_boolean('useAdam', False,"""Use adam optimizer instead""")
tf.app.flags.DEFINE_integer('colditer', 50,"""number of cold start iteration using SGD""")
###################
tf.app.flags.DEFINE_integer('batchSize', 1024,"""mini-batch size""")
tf.app.flags.DEFINE_integer('test_batchsize', 5000 ,"""test mini-batch   size""")
tf.app.flags.DEFINE_integer('randomseed', 1234 ,"""random seed""")
tf.app.flags.DEFINE_integer('maxiter', 100000 ,"""maximum updates""")
tf.app.flags.DEFINE_integer('validiter', 0 ,"""validation updates""")

tf.set_random_seed(FLAGS.randomseed)
# print the script flags
print '##################'
for key in FLAGS.__dict__['__flags']:
  print key, ': ', FLAGS.__dict__['__flags'][key]
print '##################'

BATCHSIZE = FLAGS.batchSize
FAC_UPDATE = FLAGS.colditer
kfac.KFAC_DEBUG = FLAGS.debug

LR = FLAGS.lr
MOM = FLAGS.mom
EPSILON = FLAGS.eps
STATS_DECAY = FLAGS.statDecay
ASYNC_KFAC = FLAGS.useAsync

validUpdate = FLAGS.validiter
if FLAGS.validiter <= 0:
    validUpdate = int(float(55000)/BATCHSIZE)

MAX_ITER = FLAGS.maxiter
NUM_THREADS = 16  # maybe not working? somehow always only use 5 threads :/

import math
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUMHIDDEN = 32




def fill_feed_dict(data_set, images_pl, labels_pl):

  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_batch(BATCHSIZE)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

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
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def NLLLoss(logits, labelsPL):

  batch_size = tf.size(labelsPL)
  labels = tf.expand_dims(labelsPL, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(axis=1, values=[indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

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
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_expanded,
                                                          labels=onehot_labels,
                                                          name='xentropy_sample')
  loss_sampled = tf.reduce_mean(cross_entropy, name='xentropy_mean_sample')
  return loss_sampled

def Linear(nIn, nOut, inputX, name):
    with tf.name_scope(name):
        weights = tf.Variable(
            tf.truncated_normal([nIn, nOut],
                                stddev=1.0 / math.sqrt(float(nIn))),
            name='weights')
        biases = tf.Variable(tf.zeros([nOut]),
                             name='biases')
        
        #return (tf.matmul(inputX, weights) + biases)
        return tf.nn.bias_add(tf.matmul(inputX, weights), biases)

def ConvPool(nIn, nOut, inputX, name, kx=5, ky=5, dx=1, dy=1):
    with tf.name_scope(name):
        kernel = tf.Variable(tf.truncated_normal([kx, ky, nIn, nOut],
                                stddev=1.0 / math.sqrt(float(nIn))),
            name='weights')
        biases = tf.Variable(tf.zeros([nOut]),
                             name='biases')
        conv = tf.nn.conv2d(inputX, kernel, [1, dx, dy, 1], padding='SAME')
        #pooled = tf.nn.max_pool(conv+biases, [1, 4, 4, 1], [1, 2, 2,1], padding='SAME')
        pooled = tf.nn.max_pool(tf.nn.bias_add(conv, biases), [1, 4, 4, 1], [1, 2, 2,1], padding='SAME')
        return pooled


def buildModel(inputPL):

  if FLAGS.useFCModel:
    # Hidden 1
    h1 = tf.nn.relu(Linear(784, 1000, inputPL, 'h1'))
    # Hidden 2
    h2 = tf.nn.relu(Linear(1000, 1000, h1, 'h2'))
    # Output
    out = Linear(1000, NUM_CLASSES, h2, 'out')

  else:
    inputPL = tf.reshape(inputPL, [BATCHSIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    # Hidden 1
    h1 = tf.nn.relu(ConvPool(1, NUMHIDDEN, inputPL, 'h1'))
    # Hidden 2
    h2 = tf.nn.relu(ConvPool(NUMHIDDEN, NUMHIDDEN, h1, 'h2'))
    # Output
    h2 = tf.reshape(h2, [BATCHSIZE, -1])
    out = Linear(7*7*NUMHIDDEN, NUM_CLASSES, h2, 'out')
    
  return out

with tf.device('/gpu'):
   inputPL = tf.placeholder(tf.float32, shape=(BATCHSIZE, IMAGE_PIXELS)) 
   labelsPL = tf.placeholder(tf.int32, shape=(BATCHSIZE)) 
   
   numUpdates = tf.placeholder(tf.int32)
   modelOut = buildModel(inputPL)
   loss = NLLLoss(modelOut, labelsPL)
   loss_sampled = NLLLoss_sampled(modelOut)
    
   acc = evaluation(modelOut, labelsPL)

   queue_runner = None
   if FLAGS.useAdam:
      updates = tf.train.AdamOptimizer().minimize(loss)
   else:
      updates, queue_runner = kfac.AKfacOptimizer(learning_rate=LR, momentum=MOM, epsilon=EPSILON, stats_decay=STATS_DECAY, async=ASYNC_KFAC, cold_iter=FLAGS.colditer).minimize(loss, loss_sampled)
      #updates, queue_runner = kfac.KfacOptimizer(learning_rate=LR, momentum=MOM, epsilon=EPSILON, stats_decay=STATS_DECAY, async=ASYNC_KFAC, cold_iter=FLAGS.colditer).minimize(loss, loss_sampled)

   #step = [item for item in tf.all_variables() if item.name == 'KFAC/global_step:0'][0]



data_sets = mnist_data.read_data_sets('data')

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=NUM_THREADS)) as session:

    #tf.initialize_all_variables().run()
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #tf.local_variables_initializer().run()
    #tf.global_variables_initializer().run()

    train_acc_tol = 0
    train_loss_tol = 0
    start_time = time.time()
    for i in range(MAX_ITER):
        feed_dict = fill_feed_dict(data_sets.train,
                            inputPL,
                            labelsPL)
        feed_dict[numUpdates] = i
        _, Loss, Acc = session.run([updates, loss, acc],
                          feed_dict=feed_dict)
        if queue_runner != None and i == (FAC_UPDATE-5):
            print("###############")
            print("starting the queue")
            print("###############")
            coord = tf.train.Coordinator()
            enqueue_threads = queue_runner.create_threads(session, coord=coord, start=True)
        train_acc_tol += Acc
        train_loss_tol += Loss
        if (i+1)%validUpdate == 0:
            test_acc_tol = 0
            test_loss_tol = 0
            duration = time.time() - start_time
            num_testBatches = math.floor(10000./BATCHSIZE)
            for j in range(int(num_testBatches)):
                feed_dict = fill_feed_dict(data_sets.test,
                                     inputPL,
                                     labelsPL)
                feed_dict[numUpdates] = i
                test_loss, test_acc = session.run([loss, acc],feed_dict=feed_dict)

                test_loss_tol += test_loss
                test_acc_tol += test_acc

            train_acc_tol /= float(validUpdate*BATCHSIZE)
            test_acc_tol /= float(num_testBatches*BATCHSIZE)
            train_loss_tol /= float(validUpdate)
            test_loss_tol /= float(num_testBatches)
            print 'update : ', int(i/validUpdate)
            print 'train loss: ',train_loss_tol,' train acc: ',train_acc_tol
            print 'test loss: ', test_loss_tol, ' test acc: ', test_acc_tol
            print validUpdate*BATCHSIZE/duration, ' images per sec.'
            start_time = time.time()

            train_loss_list.append(train_loss_tol)
            test_loss_list.append(test_loss_tol)
            train_acc_list.append(train_acc_tol)
            test_acc_list.append(test_acc_tol)

            np.save(FLAGS.name+'.npy', [np.array(train_loss_list),
                                        np.array(test_loss_list),
                                        np.array(train_acc_list),
                                        np.array(test_acc_list),])
            train_acc_tol = 0
            train_loss_tol = 0
