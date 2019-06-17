import sys
sys.path.append('/home/jimmy/gobi2/projects/async_kfac/')

import tensorflow as tf
import numpy as np
import time
import mnist_data
import kfac_conv_utils
import kfac


FLAGS = tf.app.flags.FLAGS
tf.set_random_seed(123)
# print the script flags
print '##################'
for key in FLAGS.__dict__['__flags']:
  print key, ': ', FLAGS.__dict__['__flags'][key]
print '##################'


Ffactor_feat_map = tf.random_normal([100, 10, 10, 2])
conv_w = tf.random_normal([2, 2, 2, 6])
Bfactor_feat_map = tf.nn.conv2d(Ffactor_feat_map, conv_w, 
                                strides=[1, 1, 1, 1], padding='SAME')
tmp = tf.reshape(Bfactor_feat_map, [-1, 6])
gt_cov_b = tf.matmul(tmp, tmp, transpose_a=True)/100  # E[ggT]*T

conv_op = Bfactor_feat_map.op
cov_f = kfac_conv_utils.compute_conv2D_stats_default(Ffactor_feat_map, fprop=True, fops=conv_op, var=conv_w)
print(cov_f)
cov_b = kfac_conv_utils.compute_conv2D_stats_default(Bfactor_feat_map, bprop=True, fops=conv_op, var=conv_w)
print(cov_b)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=16)) as session:
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    _cov_f, _cov_b, _gt_cov_b = session.run([cov_f, cov_b, gt_cov_b])


    np.set_printoptions(precision=2)
    print(_cov_f)
    print(_cov_b)
    print(_gt_cov_b)
    
