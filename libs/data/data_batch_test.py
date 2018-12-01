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
import scipy.misc
from libs.data.data_batch import Data_Loader
import cv2

def DataLoaderTest():
    # def test_get_batch(self):
    g = tf.Graph()
    with g.as_default():
        with tf.device('/cpu:0'):
            data_loader = Data_Loader(dataset_dir='test',
                                      split_sizes='train')
            data_loader.get_dataset()
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                j = 0
                try:
                    while not coord.should_stop() and j < 2:
                        image, seg_maps = data_loader.get_batch()
                        print('image shape:', image.shape)
                        print('seg_maps shape:', seg_maps.shape)
                        print(seg_maps)
                        g_image,  g_seg = sess.run([image, seg_maps])
                        for i, img in enumerate(g_image):
                            print(img.shape)
                            cv2.imwrite('test/{}_{}_ori.png'.format(j, i), img)
                        for i , img in enumerate(g_seg):
                            print(img.shape)
                            for z in range(3):
                                scipy.misc.imsave('test/{}_{}_{}_seg.png'.format(j, i, z), img[:,:,z])
                        j += 1
                        print(j)
                except tf.errors.OutOfRangeError:
                    print('out of range')
                finally:
                    print('done')
                    coord.request_stop()
                coord.join(threads=threads)


if __name__ == '__main__':
    test = DataLoaderTest()