import os
import glob

import skimage
import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt

from shutil import copyfile

from shutil import copytree
from shutil import rmtree

from supreshelper import *

from typing import List
from typing import Tuple

import tensorflow as tf

import argparse


def save_scene(scene_path, normalize_lrs:bool, normalize_hr: bool, top_k):
    
    # todo make param
    data_destination = "DataNormalized"
    
    new_path = data_destination + "/" + "/".join(scene_path.split("/")[1:])
    os.makedirs(new_path, exist_ok=True)
    
    if not top_k:
        lr_filler = central_tendency(scene_path, agg_with='median', only_clear=False)
        if normalize_lrs:
            lr_filler = normalize(lr_filler)
        
        lrs = load_and_normalize_lrs(scene_path, normalize_lrs)

        # Fill the list with the "filler" LR to get 35 filters
        for i in range(0, max_lrs - len(lrs)):
            lrs.append(lr_filler)

        # Shuffle the list randomly
        random.shuffle(lrs)
        
    else:
        # Top K filling
        lrs_path = sorted(glob(scene_path + '/LR*.png'))
        qms_path = sorted(glob(scene_path + '/QM*.png'))
        
        lr_qms = [(skimage.io.imread(lr), skimage.io.imread(qm)) for lr, qm in zip(lrs_path, qms_path)]
        # (qm score, lr as float, qm in [0,1])
        lr_qms = [(np.sum(qm), skimage.img_as_float64(lr << 2), qm / 255.0) for lr, qm in lr_qms]
        # qm_score, lr with concealed pixel as 0
        lr_qms = [(qm_score, lr * qm) for qm_score, lr, qm in lr_qms]
        # sorted by qm_score
        lr_qms = sorted(lr_qms, key = lambda t: t[0], reverse=True)
        
        lrs = []
        idx = 0
        
        # cycle through lr_qms until we have max_lrs of them
        while (len(lrs) < 35):
            if normalize_lrs:
                lrs.append(normalize(lr_qms[idx][1]))
            else:
                lrs.append(lr_qms[idx][1])
                
            idx += 1
            idx = idx % len(lr_qms)
            
        # Shuffle LRS?
        # random.shuffle(lrs)
        
    sm = skimage.img_as_float64(skimage.io.imread(scene_path + "/" + 'SM.png'))
    
    if scene_path.split("/")[1] == "train":
        hr = load_and_normalize_hr(scene_path, normalize_hr)
        hr = np.where(sm > 0, hr, np.NaN)
        array_to_tfrecords_multiple(lrs, hr, new_path + "/" + "multiple.tfrecords")
        
    else:
        array_to_tfrecords_multiple(lrs, np.ones([384,384]), new_path + "/" + "multiple.tfrecords")
    
    for f in glob(scene_path+ "/*"):
        copyfile(f,  data_destination + "/" + "/".join(f.split("/")[1:]))

def array_to_tfrecords_multiple(lrs, hr, output_file):
    feature ={"lrs": tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate(lrs).ravel())),
              # We don't need the list length after deciding to use a fixed length now
              #"lrs_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(lrs)])),
              "hr": tf.train.Feature(float_list=tf.train.FloatList(value=hr.flatten())),
             }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()

    writer = tf.io.TFRecordWriter(output_file)
    writer.write(serialized)
    writer.close()
    

def mergeAndUpscale(scene_path):
    merged = central_tendency(scene_path, agg_with='median', only_clear=False)
    return bicubic_upscaling(merged)


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def load_and_normalize_hr(scene_path, normalize=False):
    hr, _ = highres_image(scene_path, img_as_float=False)
    hr = skimage.img_as_float64(hr << 2)
    if normalize:
        return normalize(hr)
    else:
        return hr

def load_and_normalize_lrs(scene_path):
    
    normalized_lrs = []
    for lr, _ in lowres_image_iterator(scene_path, img_as_float=False):
        lr = skimage.img_as_float64(lr << 2)
        normalized_lrs.append(normalize(lr))
    
    return normalized_lrs


def main(data_path: str, normalize_lrs: bool,  normalize_hr: bool, top_k: bool):

    print("Getting data from " + data_path)
    print("normalize lr: " + str(normalize_lrs))
    print("normalize hr: " + str(normalize_hr))
    print("top_k LRs (recommended tbd): " + str(top_k))
    
    train = glob(data_path + "train/*/*") 
    test = glob(data_path + "test/*/*") 
    
    print("number of train elements : " + str(len(train)))
    print("number of test elements : " + str(len(test)))
    
    # Count max number of LRs in all scenes
    max_lrs = 0
    for scene in glob(data_path + "train/*/*") + glob(data_path + "test/*/*"):
        lrs = len(glob(scene + "/LR*"))
        if max_lrs < lrs:
            max_lrs = lrs
            
    for scene in train + test:
        #print("saving scene " + scene)
        save_scene(scene, normalize_lrs, normalize_hr, top_k)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate TFRecords from the ProbaV Data")
    parser.add_argument("--data_dir", help="path to the directory containing the data", type=str, default="Data/")
    parser.add_argument("-norm_lrs", "--normalize_lrs", help="enable LRs normalization recommended", action="store_true")
    parser.add_argument("-norm_hr", "--normalize_hr", help="enabling HR normalization (not recommneded)", action="store_true")
    parser.add_argument("--top_k", help="Enable topK LR filling instead of mean filling", action="store_true")
    
    # shuffle LR
    
    args = parser.parse_args()
    main(args.data_dir, args.normalize_lrs, args.normalize_hr, args.top_k)
    
    