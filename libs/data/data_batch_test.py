# -*- coding: utf-8 -*-
# @Time    : 2018/11/27 10:16
# @Author  : zsz
# @Site    : 
# @File    : data_batch_test
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf

from libs.data.data_batch import Data_Loader
import cv2

def DataLoaderTest():
    # def test_get_batch(self):
    g = tf.Graph()
    with g.as_default():
        with tf.device('cpu:0'):
            data_loader = Data_Loader(dataset_dir='test',
                                      split_sizes={'train': 200})
            data_loader.get_dataset()
            with tf.Session() as sess:
                image, seg_maps = data_loader.get_batch()
                print('segmaps shape:', image.shape)
                print('labels:', seg_maps.shape)
                g_img, g_seg = sess.run([image, seg_maps])

                for i, img in enumerate(g_img):
                    cv2.imwrite('test/{}_ori.png'.format(i))


if __name__ == '__main__':
    test = DataLoaderTest()