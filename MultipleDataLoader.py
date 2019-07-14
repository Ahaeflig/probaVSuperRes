import os
from glob import glob

import skimage
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from supreshelper import *

from typing import List
import time

LR_HR_RATIO = 3

class MultipleDataLoader():
    """ Class that provdes utilities to read and process TFRecords that contains multiple LR images and their correpsonding HR image. The main point of the class is to map a tf.data.TFRecordDataset to the parse function. This will turn the TF dataset into an iterator yields Tensor ready to be fed to the model.
    """
    
    def __init__(self, data_dir: str):
        pass

    @tf.function
    def parse_multiple_fixed(self, example_proto, augment=False):
        """ Parses a tf record and reconstruct the correct data from the following features:
        
            feature ={"lrs": tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate(lrs).ravel())),
                      "hr": tf.train.Feature(float_list=tf.train.FloatList(value=hr.flatten()))}
            
            where lrs always has 35 lr images
        """
        keys_to_features = {'lrs':tf.io.FixedLenFeature((35, 128, 128), tf.float32),
                            'hr': tf.io.FixedLenFeature((384, 384), tf.float32)}
        
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        
        lrs_merged = parsed_features['lrs']
        lrs_reshaped = tf.transpose(lrs_merged, perm=[1,2,0])
        
        if augment:
            return self.augment(lrs_reshaped, tf.expand_dims(parsed_features['hr'], axis=2))
        else:
             return lrs_reshaped, tf.expand_dims(parsed_features['hr'], axis=2)
        
        
    @tf.function
    def parse_multiple_flexible(self, example_proto, augment=False):
        """ Parses a tf record and reconstruct the correct data from the following features:
        
            feature ={"lrs": tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate(lrs).ravel())),
              "lrs_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(lrs)])),
              "hr": tf.train.Feature(float_list=tf.train.FloatList(value=hr.flatten())),
             }
             
             where lrs can have a flexible amount of lr images
        """
        keys_to_features = {'lrs':tf.io.VarLenFeature(tf.float32),
                            'lrs_length': tf.io.FixedLenFeature((1), tf.int64),
                            'hr': tf.io.FixedLenFeature((384,384), tf.float32)}
        
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        
        lrs_merged = tf.sparse.to_dense(parsed_features['lrs'])
        lrs_length = parsed_features['lrs_length']
        
        lrs_reshaped = tf.reshape(lrs_merged, (-1, 128,128))
        
        # We take 9 LRs from all stored LRs, minimum among all scenes
        lrs_reshaped = tf.random.shuffle(lrs_reshaped)[:9,:,:]
        
        lrs_reshaped = tf.transpose(lrs_reshaped, perm=[1,2,0])
        
        if augment:
            return self.augment(lrs_reshaped, tf.expand_dims(parsed_features["hr"], axis=2))
        else:
            lrs_reshaped, tf.expand_dims(parsed_features["hr"], axis=2)
        
    @tf.function
    def augment(self, lrs, hr):
        
        if tf.random.uniform(()) >= 0.5:
            lrs = tf.image.flip_left_right(lrs)
            hr = tf.image.flip_left_right(hr)
            
        if tf.random.uniform(()) >= 0.5:
            lrs = tf.image.flip_up_down(lrs)
            hr = tf.image.flip_up_down(hr)
        
        return lrs, hr
            