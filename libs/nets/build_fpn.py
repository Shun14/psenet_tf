# -*- coding: utf-8 -*-
# @Time    : 2018/8/17 11:35
# @Author  : zsz
# @Site    : 
# @File    : build_fpn.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.nets.slim_nets import resnet_v1
from configs.train_config import TRIAN_CONFIG

def unpool(inputs, name,factor=2):
    return tf.image.resize_bilinear(inputs, [tf.shape(inputs)[1]*int(factor), tf.shape(inputs)[2]*int(factor)], name=name)

class FPN(object):
    def __init__(self,
                 net_name,
                 inputs,
                 is_training=False,
                 rpn_weight_decay=0.0001,
                 ):
        self.net_name = net_name
        self.flags = self.get_flags_byname(self.net_name)

        self.logits , self.share_net = self.get_network_byname(is_training = is_training, inputs=inputs)
        self.rpn_weight_decay = rpn_weight_decay

        self.feature_maps_dict = self.get_feature_maps()
        self.fpn = self.build_fpn()

        self.pre_seg_maps = self.build_PSENET()
        self.gt_seg_maps = None

    def get_logits_and_share_net(self):
        return self.logits, self.share_net

    def get_flags_byname(self, net_name):
        if net_name not in ['resnet_v1_50', 'mobilenet_224', 'inception_resnet',
                            'vgg16', 'resnet_v1_101']:
            raise ValueError(
                "not include network: {}, we allow resnet_v1_50, mobilenet_224, inception_resnet, "
                "vgg16, resnet_v1_101"
                "")

        if net_name == 'resnet_v1_50':
            from configs import config_resnet_50
            return config_resnet_50.FLAGS
        if net_name == 'mobilenet_224':
            pass
            # from configs import config_mobilenet_224
            # return config_mobilenet_224.FLAGS
        if net_name == 'inception_resnet':
            pass
            # from configs import config_inception_resnet
            # return config_inception_resnet.FLAGS
        if net_name == 'vgg16':
            pass
            # from configs import config_vgg16
            # return config_vgg16.FLAGS
        if net_name == 'resnet_v1_101':
            from configs import config_res101
            return config_res101.FLAGS


    def get_network_byname(self,
                           inputs,
                           num_classes=1000,
                           is_training=True,
                           global_pool=True,
                           output_stride=None,
                           spatial_squeeze=True):
        net_name= self.net_name
        FLAGS = self.flags
        if net_name == 'resnet_v1_50':
            with slim.arg_scope(resnet_v1.resnet_arg_scope(
                    weight_decay=FLAGS.weight_decay)):
                logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                            num_classes=num_classes,
                                                            is_training=is_training,
                                                            global_pool=global_pool,
                                                            output_stride=output_stride,
                                                            spatial_squeeze=spatial_squeeze
                                                            )

            return logits, end_points
        if net_name == 'resnet_v1_101':
            with slim.arg_scope(resnet_v1.resnet_arg_scope(
                    weight_decay=FLAGS.weight_decay)):
                logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                             num_classes=num_classes,
                                                             is_training=is_training,
                                                             global_pool=global_pool,
                                                             output_stride=output_stride,
                                                             spatial_squeeze=spatial_squeeze
                                                             )
            return logits, end_points

    def get_feature_maps(self):

        '''
            Compared to https://github.com/KaimingHe/deep-residual-networks, the implementation of resnet_50 in slim
            subsample the output activations in the last residual unit of each block,
            instead of subsampling the input activations in the first residual unit of each block.
            The two implementations give identical results but the implementation of slim is more memory efficient.

            SO, when we build feature_pyramid, we should modify the value of 'C_*' to get correct spatial size feature maps.
            :return: feature maps
        '''

        with tf.variable_scope('get_feature_maps'):
            if self.net_name == 'resnet_v1_50':
                feature_maps_dict = {
                    'C2': self.share_net[
                        'resnet_v1_50/block1'],  # [56, 56]
                    'C3': self.share_net[
                        'resnet_v1_50/block2'],  # [28, 28]
                    'C4': self.share_net[
                        'resnet_v1_50/block3'],  # [14, 14]
                    'C5': self.share_net['resnet_v1_50/block4']  # [7, 7]
                }
            elif self.net_name == 'resnet_v1_101':
                feature_maps_dict = {
                    'C2': self.share_net[
                        'resnet_v1_101/block1'],
                # [56, 56]
                    'C3': self.share_net[
                        'resnet_v1_101/block2'],
                # [28, 28]
                    'C4': self.share_net[
                        'resnet_v1_101/block3'],
                # [14, 14]
                    'C5': self.share_net['resnet_v1_101/block4']  # [7, 7]
                }
            else:
                raise Exception('get no feature maps')

            return feature_maps_dict

    def build_fpn(self):
        '''
        build P2, P3, P4, P5
        :return: multi scale feature map
        '''
        feature_pyramid = {}
        with tf.variable_scope('build_fpn'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
                feature_pyramid['P5'] = slim.conv2d(self.feature_maps_dict['C5'],num_outputs=256,kernel_size=[1, 1],stride=1,scope='build_P5')
                feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],kernel_size=[2, 2],stride=2,scope='build_P6')

                # P6 is down sample of P5
                for layer in range(4, 1, -1):
                    p, c = feature_pyramid['P' + str(layer + 1)], self.feature_maps_dict['C' + str(layer)]
                    up_sample = unpool(p, 'build_P%d/up_sample_nearest_neighbor' % layer, 2)
                    c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1, scope='build_P%d/reduce_dimension' % layer)
                    # add or dot
                    p = up_sample + c
                    p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1, padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                    feature_pyramid['P' + str(layer)] = p

        return feature_pyramid

    def build_PSENET(self):
        '''
        :return: seg_feature_maps format (batch_size, train_scale, train_scale, number_of_kernels)
        '''
        n = TRIAN_CONFIG['number_kernel_scales']
         # TRIAN_CONFIG[]
        fpn_net = self.fpn
        seg_feature_maps_list = []
        with tf.variable_scope('build_PSENET'):

            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):

                fpn_p2_combine = tf.concat([fpn_net['P2'],  unpool(fpn_net['P3'], 'unpool_p3', 2)], axis=-1)
                fpn_p3_combine = tf.concat([fpn_p2_combine, unpool(fpn_net['P4'], 'unpool_p4', 4)], axis=-1)
                fpn_p4_combine = tf.concat([fpn_p3_combine, unpool(fpn_net['P5'], 'unpool_p5', 8)], axis=-1)
                #build fusion feature
                fpn_end = slim.conv2d(fpn_p4_combine, 256, [3,3], stride=1, activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm, scope='psenet_head')

                for i in range(n):
                    seg_map = slim.conv2d(fpn_end, 1, [1,1], normalizer_fn=None, activation_fn=None)
                    up_seg_map = tf.image.resize_bilinear(seg_map, size=[tf.shape(seg_map)[1]*int(4),  tf.shape(seg_map)[2]*int(4)])
                    seg_map = tf.sigmoid(up_seg_map)
                    seg_map = tf.Print(seg_map, [tf.shape(seg_map), seg_map], message='seg map:', summarize=20)
                    seg_feature_maps_list.append(seg_map)
        seg_feature_maps = tf.concat(seg_feature_maps_list, axis=-1)
        return seg_feature_maps


    def build_loss(self, gt_seg_maps):
        #gt seg maps format (batch_size, train_scale, train_scale, number_of_kernels) same as pre_seg_maps
        pre_seg_maps = self.pre_seg_maps
        n = TRIAN_CONFIG['number_kernel_scales']
        lambda_train = TRIAN_CONFIG['lambda_train']
        loss = None
        with tf.name_scope('Loss'):
        # for i in range(n):
            tf.add_to_collection('losses', loss)

        tf.add_n(tf.get_collection('losses'), name='total_loss')



if __name__ == '__main__':
    # test fpn class output
    import os
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_input = tf.Variable(initial_value=tf.ones((((2, 640,640,3)))),dtype=tf.float32)

    fpn_model = FPN('resnet_v1_101',test_input, is_training=True)
    # output = fpn_model.model()
    output = fpn_model.pre_seg_maps
    init_op = tf.global_variables_initializer()
    restore = slim.assign_from_checkpoint_fn('libs\\nets\\resnet_v1_101\\resnet_v1_101.ckpt', slim.get_trainable_variables(), ignore_missing_vars=True)

    logits, share_net = fpn_model.get_logits_and_share_net()

    feature_maps = fpn_model.get_feature_maps()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(init_op)
        restore(sess)
        # out  = sess.run([output])
        # print(len(out[0]))
        logit_print = sess.run([logits])
        feature_maps_print = sess.run([feature_maps])
        print('***************logits*****************')
        print(np.array(logit_print).shape)
        print('************share_net************')
        # print(feature_maps_print.keys())
        for map in feature_maps_print:
            print(map.keys())
        # share_net_print = sess.run([share_net])
        # print(np.array(share_net_print).shape)
        keys = share_net.keys()
        # print(keys)
        with open('keys.txt', 'w', encoding='utf-8') as f:
            for key in keys:
                f.write('{} {}\n'.format(key, share_net[key].shape))
