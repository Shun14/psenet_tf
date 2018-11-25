# -*- coding: utf-8 -*-
# @Time    : 2018/8/30 21:59
# @Author  : zsz
# @Site    : 
# @File    : data_batch
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import os
import cv2
import tensorflow.contrib.slim as slim
from configs.train_config import TRIAN_CONFIG
from libs.processing import ssd_vgg_preprocessing

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}


def get_datasets(data_dir,split_sizes, file_pattern = '*.tfrecord'):
    file_patterns = os.path.join(data_dir, file_pattern)
    print('file_path: {}'.format(file_patterns))
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ignored': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/oriented_bbox/x1': slim.tfexample_decoder.Tensor('image/object/bbox/x1'),
        'object/oriented_bbox/x2': slim.tfexample_decoder.Tensor('image/object/bbox/x2'),
        'object/oriented_bbox/x3': slim.tfexample_decoder.Tensor('image/object/bbox/x3'),
        'object/oriented_bbox/x4': slim.tfexample_decoder.Tensor('image/object/bbox/x4'),
        'object/oriented_bbox/y1': slim.tfexample_decoder.Tensor('image/object/bbox/y1'),
        'object/oriented_bbox/y2': slim.tfexample_decoder.Tensor('image/object/bbox/y2'),
        'object/oriented_bbox/y3': slim.tfexample_decoder.Tensor('image/object/bbox/y3'),
        'object/oriented_bbox/y4': slim.tfexample_decoder.Tensor('image/object/bbox/y4'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/ignored': slim.tfexample_decoder.Tensor('image/object/bbox/ignored')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {0:'background', 1:'text'}


    return slim.dataset.Dataset(
        data_sources=file_patterns,
        reader=reader,
        decoder=decoder,
        num_samples=split_sizes['train'],
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=TRIAN_CONFIG['num_classes'],
        labels_to_names=labels_to_names)

class Data_Loader(object):
    def __init__(self, dataset_dir, split_sizes={'train': 2518}):
        self.queue = None
        self.dataset_dir = dataset_dir
        self.split_sizes = split_sizes

    def get_dataset(self, split_sizes):
        datasets = get_datasets(self.dataset_dir, self.split_sizes)
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    datasets,
                    num_readers=2,
                    common_queue_capacity=200 * 10,
                    common_queue_min=10 * 100,
                    shuffle=True)

        [gimage, glabels, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get(['image',
                                                             'object/label',
                                                             'object/bbox',
                                                             'object/oriented_bbox/x1',
                                                             'object/oriented_bbox/x2',
                                                             'object/oriented_bbox/x3',
                                                             'object/oriented_bbox/x4',
                                                             'object/oriented_bbox/y1',
                                                             'object/oriented_bbox/y2',
                                                             'object/oriented_bbox/y3',
                                                             'object/oriented_bbox/y4'
                                                             ])
        
        gxs = tf.transpose(tf.stack([x1, x2, x3, x4])) #shape = (N,4)
        gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
        #after preprocessing get seg maps
        image, labels, bboxes, xs, ys = ssd_vgg_preprocessing.preprocess_image(gimage, glabels, gbboxes, gxs, gys, out_shape=TRIAN_CONFIG['train_scale'] ,is_training=True)
        #get seg maps and labels
        image = tf.identity(image, 'processed_image')
        seg_maps, labels = self.get_seg_maps(xs, ys, bboxes, labels, mode=TRIAN_CONFIG['dataset_format']['H'])
        #create batches and queue
        with tf.name_scope('batches'):
            b_image, b_segmaps = tf.train.batch([image, seg_maps], batch_size=TRIAN_CONFIG['batch_size'], num_threads=TRIAN_CONFIG['num_preprocessing_threads'], capacity=5* TRIAN_CONFIG['batch_size'])
        with tf.name_scope('prefetch_data'):
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_image, b_segmaps], num_threads=4, capacity=5* TRIAN_CONFIG['batch_size']
            )
        self.queue = batch_queue
    
    def get_batch(self):
        return self.queue.dequeue()

    def get_seg_maps(self, xs, ys, bboxes, labels, mode=TRIAN_CONFIG['dataset_format']['H']):
        """
        inputs:
        xs store the xs cor it's shape: (N,4) format x1, x2,x3, x4
        ys store the ys cor it's shape: (N,4) format y1, y2,y3, y4
        bboxes store the horizontal xys format :ymin, xmin, ymax, xmax
        mode: H means use bboxes for horizontal box
              P means use xs ys for quadrilateral box
        return:
        seg_maps, labels
        """
        if mode == TRIAN_CONFIG['dataset_format']['H'] and bboxes is not None:
            seg_maps , labels = tf.py_func(self.get_seg_maps_for_h_func, [bboxes, labels], [tf.float32, tf.bool])
        elif mode == TRIAN_CONFIG['dataset_format']['P'] and xs is not None and ys is not None:
            seg_maps , labels = tf.py_func(self.get_seg_maps_for_p_func, [xs, ys, labels], [tf.float32, tf.bool])
        
        return seg_maps, labels

    def get_seg_maps_for_h_func(self, bboxes, labels):
        """
        the function creates seg maps for horizontal box
        inputs:
        bboxes store the horizontal xys format :ymin, xmin, ymax, xmax
        return:
        seg_maps, labels
        """
        n = TRIAN_CONFIG['number_kernel_scales']
        train_scale = TRIAN_CONFIG['train_scale']
        m = TRIAN_CONFIG['minimal_scale_ratio']
        seg_maps = []
        
        for i in range(n, 0 , -1):
            seg_map = np.zeros([train_scale, train_scale])
            for index ,bbox in enumerate(bboxes):
                if labels[index] == 0:
                    continue
                ymin = int(bbox[0]* train_scale)
                xmin = int(bbox[1]* train_scale)
                ymax = int(bbox[2]* train_scale)
                xmax = int(bbox[3]* train_scale)
                if i == n:
                    seg_map = seg_map[ymin:ymax, xmin:xmax]
                else:
                    r_i = 1. - (float(1. -m) * (n - i)) / (n - 1) 
                    area = (ymax - ymin) * (xmax - xmin)
                    perimeter =2* (ymax - ymin + xmax - xmin)
                    d_i = area * ( 1. - r_i*r_i)
                    ymin_new = ymin + d_i
                    ymax_new = ymax - d_i
                    seg_map = seg_map[ymin_new:ymax_new, xmin:xmax]
            seg_maps.append(seg_map)
        return seg_maps, True
            


    def get_seg_maps_for_p_func(self, xs, ys, labels):
        """
        the function creates seg maps for quadrilateral box
        inputs:
        xs store the xs cor it's shape: (N,4) format x1, x2,x3, x4
        ys store the ys cor it's shape: (N,4) format y1, y2,y3, y4
        return:
        seg_maps, labels
        """
        pass

