# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 21:29
# @Author  : zsz
# @Site    : 
# @File    : train.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import libs.nets.network_factory
import libs.data.data_batch as data_batch
import libs.nets.build_fpn as fpn
from configs.train_config import TRIAN_CONFIG
import tensorflow.contrib.slim as slim
import tensorflow as tf
import os
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.85, 'GPU memory fraction to use.')

tf.app.flags.DEFINE_string(
    'train_dir', './dssd_tfmodel/synth_model/dssd_resnet',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/zsz/datasets/synth-tf/', 'The directory where the dataset files are stored.')
    # 'dataset_dir', './tfrecord_for_train/huawei_ano_with_difficult_v2', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'gpu_num', 2, 'the number of gpu use'
)
# =========================================================================== #
tf.app.flags.DEFINE_string(
    #'checkpoint_path', 'huawei_model'
    'checkpoint_path', 'resnet_model/resnet_v1_101.ckpt',
    # 'checkpoint_path', './ssd_model',
    # 'checkpoint_path', './dssd_tfmodel/synth_model/huawei_synth_v3_dssd_final_model/',#retrain a new model
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_bool(
    'ignore_missing_vars', False,
    'The parameter which means ignore missing vars in the checkpoint'
)

FLAGS = tf.app.flags.FLAGS


def tower_loss(scope, images, segmaps):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    
    # Build inference Graph.
    # FPN = fpn.FPN()
    # fpn_net = FPN.build_fpn()
    # print(fpn_net)
    fpn_model = fpn.FPN(TRIAN_CONFIG['net_name'], images, is_training=True)
    loss = fpn_model.build_loss(segmaps)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % 'psenet_tower', '', l.op.name)
        tf.summary.scalar(loss_name, l)
    return total_loss


def average_gradients(tower_grads):
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

def train():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        with tf.name_scope('data_loader'):
            with tf.device('/cpu:0'):
                data_loader = data_batch.Data_Loader(dataset_dir=FLAGS.dataset_dir, split_sizes='train')
                data_loader.get_dataset()

        with tf.device('/cpu:0'):
            global_step = tf.train.create_global_step()
            boundaries = TRIAN_CONFIG['step_boundaries']
            learning_rate = TRIAN_CONFIG['learning_rate']
            tf.summary.scalar('learning_rate', learning_rate)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, learning_rate )
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        #add clones and calculate loss
        tower_grads = []
        summaries = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(0, FLAGS.gpu_num):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('psenet_tower' , i)) as scope:
                        #dequeue
                        g_image, g_segmaps = data_loader.get_batch()
                        loss = tower_loss(scope, g_image, g_segmaps)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope.reuse_variables()

                        # Retain the summaries from the final tower
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
        
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        
        # Apply the gradients to adjust the shared variables
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(TRIAN_CONFIG['MOVING_AVERAGE_DECAY'], global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        # Group all updates to into a single train op.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([variable_averages_op, update_ops]):
            train_op = tf.group(apply_gradient_op, variable_averages)

        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               tf.get_default_graph())

        init = tf.global_variables_initializer()


        tf_config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=tf_config) as sess:
            sess.run(init)
            if FLAGS.checkpoint_path is not None:
                print('from pevious checkpoint')
                ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
                saver.restore(sess, ckpt)
            else:
                if FLAGS.pretrained_model_path is not None:
                    variable_restore_op = slim.assign_from_checkpoint_fn(
                        FLAGS.checkpoint_path, slim.get_trainable_variables(),
                    ignore_missing_vars=FLAGS.ignore_missing_vars)
                    print(' pretrained model exists')
                    variable_restore_op(sess)
                    # start_step = global_step.eval() + 1

            start_time = time.time()





if __name__ == '__main__':
    train()