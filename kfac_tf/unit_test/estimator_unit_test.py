import sys
sys.path.append('/home/jimmy/gobi2/projects/async_kfac/')

import tensorflow as tf
import numpy as np
import time
import mnist_data
import cov_estimator


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('name', 'debug',"""experiment name""")
tf.app.flags.DEFINE_boolean('debug', False,"""turn on debug flag""")
tf.app.flags.DEFINE_boolean('useFCModel', False,"""Use fully connected model""")
tf.app.flags.DEFINE_boolean('useNoisyGrad', False,"""Use noisy grad""")
tf.app.flags.DEFINE_boolean('useNoisyAdam', False,"""Use noisy adam""")
tf.app.flags.DEFINE_boolean('usePerturb', False,"""Use noisy perturbation""")
tf.app.flags.DEFINE_boolean('useDiffNoise', False,"""Use diff noise""")
tf.app.flags.DEFINE_boolean('useDiagNoise', False,"""Use diag noise""")
tf.app.flags.DEFINE_boolean('useFlipout', False,"""Use flipout""")
tf.app.flags.DEFINE_float('noise_std', 0.2,"""noise std""")
######### OPTIMIZER PARAM ##########
tf.app.flags.DEFINE_float('lr', 0.001,"""Learning rate""")
tf.app.flags.DEFINE_float('mom', 0.9,"""Momentum""")
tf.app.flags.DEFINE_boolean('useAdam', False,"""Use adam optimizer instead""")
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

LR = FLAGS.lr
MOM = FLAGS.mom

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
NUMHIDDEN = 2




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
    with tf.variable_scope(name):
        weights = tf.get_variable(
            name='weights',
            initializer=tf.truncated_normal([nIn, nOut],
                                stddev=1.0 / math.sqrt(float(nIn))),
        )
        biases = tf.get_variable(
            name='biases',
            initializer=tf.zeros([nOut]),
        )
    return tf.nn.bias_add(tf.matmul(inputX, weights), biases)

def flipout_matmul(x, w_noise, b_noise=None):
  e1 = tf.to_float(tf.random_normal(tf.shape(x))>0.)
  perturb = tf.matmul(x*e1, w_noise)
  e2 = tf.to_float(tf.random_normal(tf.shape(perturb))>0.)
  if b_noise is not None:
      perturb += b_noise
  perturb *= e2
  return perturb

def LinearPerturb(nIn, nOut, inputX, noise_dict, name):
    with tf.variable_scope(name):
        weights = tf.get_variable(
            name='weights',
            initializer=tf.truncated_normal([nIn, nOut],
                                stddev=1.0 / math.sqrt(float(nIn))),
        )
        biases = tf.get_variable(
            name='biases',
            initializer=tf.zeros([nOut]),
        )
    if not FLAGS.useFlipout:
      weights += noise_dict[weights.name]
      biases += noise_dict[biases.name]
    proj = tf.nn.bias_add(tf.matmul(inputX, weights),biases)
    if not FLAGS.useFlipout:
      return proj
    else:  
      flipout_noise = flipout_matmul(inputX, noise_dict[weights.name], noise_dict[biases.name])
      return proj + flipout_noise

def ConvPool(nIn, nOut, inputX, name, kx=5, ky=5, dx=1, dy=1):
    with tf.variable_scope(name):
        kernel = tf.get_variable(initializer=tf.truncated_normal([kx, ky, nIn, nOut],
                                stddev=1.0 / math.sqrt(float(nIn))),
            name='weights')
        biases = tf.get_variable(initializer=tf.zeros([nOut]),
                             name='biases')
        conv = tf.nn.conv2d(inputX, kernel, [1, dx, dy, 1], padding='SAME')
        #pooled = tf.nn.max_pool(conv+biases, [1, 4, 4, 1], [1, 2, 2,1], padding='SAME')
        pooled = tf.nn.max_pool(tf.nn.bias_add(conv, biases), [1, 4, 4, 1], [1, 2, 2,1], padding='SAME')
        return pooled


def buildModel(inputPL, noise_dict=None):

  if FLAGS.useFCModel:
    if noise_dict is None:
      # Hidden 1
      h1 = tf.nn.relu(Linear(784, 5, inputPL, 'h1'))
      # Hidden 2
      h2 = tf.nn.relu(Linear(5, 5, h1, 'h2'))
      # Output
      out = Linear(5, NUM_CLASSES, h2, 'out')
    else:
      # Hidden 1
      h1 = tf.nn.relu(LinearPerturb(784, 5, inputPL, noise_dict, 'h1'))
      # Hidden 2
      h2 = tf.nn.relu(LinearPerturb(5, 5, h1, noise_dict, 'h2'))
      # Output
      out = LinearPerturb(5, NUM_CLASSES, h2, noise_dict, 'out')
  else:
    inputPL = tf.reshape(inputPL, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    # Hidden 1
    h1 = tf.nn.relu(ConvPool(1, NUMHIDDEN, inputPL, 'h1'))
    # Hidden 2
    h2 = tf.nn.relu(ConvPool(NUMHIDDEN, NUMHIDDEN, h1, 'h2'))
    # Output
    h2 = tf.reshape(h2, [-1, 7*7*NUMHIDDEN])
    out = Linear(7*7*NUMHIDDEN, NUM_CLASSES, h2, 'out')
    
  return out

with tf.device('/gpu'):
   inputPL = tf.placeholder(tf.float32, shape=(BATCHSIZE, IMAGE_PIXELS)) 
   labelsPL = tf.placeholder(tf.int32, shape=(BATCHSIZE)) 
   numUpdates = tf.placeholder(tf.int32)

   with tf.variable_scope('model'):
     modelOut = buildModel(inputPL)
     loss = NLLLoss(modelOut, labelsPL)
     acc = evaluation(modelOut, labelsPL)

   vars = tf.trainable_variables()
   print([v.name for v in vars])
   
   with tf.variable_scope('model', reuse=True):
     loss1 = NLLLoss(buildModel(inputPL[:50]), labelsPL[:50])
     grads1 = tf.gradients(loss1, vars)

   with tf.variable_scope('model', reuse=True):
     loss2 = NLLLoss(buildModel(inputPL[-50:]), labelsPL[-50:])
     grads2 = tf.gradients(loss2, vars)

   diff_grads = [tf.stop_gradient(g1-g2) for g1,g2 in zip(grads1, grads2)]
   diff_grads = [tf.sign(dg)*(tf.pow(tf.abs(dg), 1.)) for dg in diff_grads]
   noise_dict = {v.name:FLAGS.noise_std*dg for v,dg in zip(vars, diff_grads)}
   if not FLAGS.useDiffNoise:
    print('use gaussian isotropic noise')
    noise_dict = {v.name:FLAGS.noise_std*tf.random_normal(tf.shape(v)) for v in vars}
   #noise_dict = {v.name: grad_noise_dict[v.name] + 0.03*tf.random_normal(tf.shape(v)) for v in vars}

   with tf.variable_scope('model', reuse=True):
     modelOut_noisy = buildModel(inputPL, noise_dict)
     loss_noisy = NLLLoss(modelOut_noisy, labelsPL)
     acc_noisy = evaluation(modelOut_noisy, labelsPL)
   if FLAGS.usePerturb:
       print("use perturbed training loss")
       train_loss = tf.Print(loss_noisy, [loss_noisy])
   else:
       train_loss = loss
   
   gs = tf.gradients(train_loss, vars, name='gradientsSampled')

   ####
   estimator = cov_estimator.Estimator(approx_option='diag')
   estimator.compute_stats(train_loss, vars, gs)

   diag_fisher_list = [estimator.factors[v]['fisherBlock_diag'] for v in vars]
   print(diag_fisher_list)

   epsilons = [FLAGS.noise_std*tf.random_normal(tf.shape(v)) for v in vars]
   fvps = estimator.FisherVectorProduct(zip(vars, epsilons), 1.)
   
   print(epsilons)
   diag_fisher_noise_prod = [tf.reshape(df, tf.shape(e))*e for df, e in zip(diag_fisher_list, epsilons)]
   
   diff_fvps = [tf.reduce_max(tf.abs(n1 - n2)) for n1, n2 in zip(fvps, diag_fisher_noise_prod)]
   
   #noise_fvp = [tf.random_normal(tf.shape(v)) for v in vars]
   #noise_fvp_sq = [n**2 for n in noise_fvp]
   #fvps_sq = estimator.FisherVectorProduct(zip(vars, noise_fvp_sq), 1.)
   #meanvp_sq = [g**2*n for g, n in zip(gs, noise_fvp_sq)]
   #fvps = [FLAGS.noise_std*tf.sign(n)*tf.sqrt(n1 - n2) for n1, n2, n in zip(fvps_sq, meanvp_sq, noise_fvp)]
   ###
   #noise_list = [noise_dict[v.name] for v in vars]

   ### debug
   #noise_list = [tf.Print(fvp,[tf.convert_to_tensor(var.name), tf.reduce_max(fvp),tf.reduce_min(fvp),tf.sqrt(tf.reduce_sum(fvp**2)), tf.reduce_max(var),tf.reduce_min(var), tf.sqrt(tf.reduce_sum(var**2))]) for var, fvp in zip(vars, noise_list)]
   #fvps = [tf.Print(fvp,[tf.convert_to_tensor(var.name), tf.reduce_max(fvp),tf.reduce_min(fvp),tf.sqrt(tf.reduce_sum(fvp**2)), tf.reduce_max(var),tf.reduce_min(var), tf.sqrt(tf.reduce_sum(var**2))]) for var, fvp in zip(vars, fvps)]
   ####



   if FLAGS.useNoisyGrad:
     gs = [g+noise_dict[v.name] for g, v in zip(gs, vars)]
   queue_runner = None
   if FLAGS.useAdam:
      print("use Adam without noise")
      updates = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(train_loss)
   else:
      print("use SGD")
      updates = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9).minimize(train_loss)


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
        _diag_fvps = session.run(diff_fvps,
                              feed_dict=feed_dict)
        print(_diag_fvps)       

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

            train_acc_tol = 0
            train_loss_tol = 0
