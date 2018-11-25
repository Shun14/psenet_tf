# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 21:28
# @Author  : zsz
# @Site    : 
# @File    : config_res101.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

######################
#    data set
####################
tf.app.flags.DEFINE_string(
    'dataset_tfrecord',
    '../data/tfrecords',
    'tfrecord of fruits dataset'
)
tf.app.flags.DEFINE_integer(
    'new_img_size',
    224,
    'the value of new height and new width, new_height = new_width'
)

###########################
#  data batch
##########################
tf.app.flags.DEFINE_integer(
    'num_classes',
    134,
    'num of classes'
)
tf.app.flags.DEFINE_integer(
    'batch_size',
    64, #64
    'num of imgs in a batch'
)
tf.app.flags.DEFINE_integer(
    'val_batch_size',
    32,
    'val or test batch'
)
###########################
## learning rate
#########################
tf.app.flags.DEFINE_float(
    'lr_begin',
    0.001, # 0.01 # 0.001 for without prepocess
    'the value of learning rate start with'
)
tf.app.flags.DEFINE_integer(
    'decay_steps',
    20000, # 5000
    "after 'decay_steps' steps, learning rate begin decay"
)
tf.app.flags.DEFINE_float(
    'decay_rate',
    0.1,
    'decay rate'
)

###############################
# optimizer-- MomentumOptimizer
###############################
tf.app.flags.DEFINE_float(
    'momentum',
    0.9,
    'accumulation = momentum * accumulation + gradient'
)

############################
#  train
########################
tf.app.flags.DEFINE_integer(
    'max_steps',
    4003,
    'max iterate steps'
)

tf.app.flags.DEFINE_string(
    'pretrained_model_path',
    '../data/pretrained_weights/resnet_v1_101.ckpt',
    'the path of pretrained weights'
)
tf.app.flags.DEFINE_float(
    'weight_decay',
    0.0001,
    'weight_decay in regulation'
)
################################
# summary and save_weights_checkpoint
##################################
tf.app.flags.DEFINE_string(
    'summary_path',
    '../output/res101_summary',
    'the path of summary write to '
)
tf.app.flags.DEFINE_string(
    'trained_checkpoint',
    '../output/res101_trained_weights',
    'the path to save trained_weights'
)
FLAGS = tf.app.flags.FLAGS