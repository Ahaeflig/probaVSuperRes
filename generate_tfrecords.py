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

import argparse


def save_scene(scene_path):
    new_path = data_destination + "/" + "/".join(scene_path.split("/")[1:])
    os.makedirs(new_path, exist_ok=True)
    
    lr_filler = central_tendency(scene_path, agg_with='median', only_clear=False)
    lr_filler = normalize(lr_filler)
    
    lrs = load_and_normalize_lrs(scene_path)

    # Fill the list with the "filler" LR to get 35 filters
    for i in range(0, max_lrs - len(lrs)):
        lrs.append(lr_filler)
    
    # Shuffle the list randomly
    random.shuffle(lrs)
    
    sm = skimage.img_as_float64(skimage.io.imread(scene_path + "/" + 'SM.png'))
    
    if scene_path.split("/")[1] == "train":
        hr = load_and_normalize_hr(scene_path)
        hr = np.where(sm > 0, hr, np.NaN)
        array_to_tfrecords_multiple(lrs, hr, new_path + "/" + "multiple.tfrecords")
        
    else:
        array_to_tfrecords_multiple(lrs, np.ones([384,384]), new_path + "/" + "multiple.tfrecords")
    
    for f in glob(scene_path+ "/*"):
        copyfile(f,  data_destination + "/" + "/".join(f.split("/")[1:]))


def array_to_tfrecords_single(all_, output_file):
    feature ={ "all": tf.train.Feature(float_list=tf.train.FloatList(value=all_.flatten()))}
    
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


def load_and_normalize_hr(scene_path):
    hr, _ = highres_image(scene_path, img_as_float=False)
    hr = skimage.img_as_float64(hr << 2)
    return normalize(hr)


def load_and_normalize_lrs(scene_path):
    
    normalized_lrs = []
    for lr, _ in lowres_image_iterator(scene_path, img_as_float=False):
        lr = skimage.img_as_float64(lr << 2)
        normalized_lrs.append(normalize(lr))
    
    return normalized_lrs


def main(data_path: str):

    data_destination = "DataNormalized"

    train = glob(data_path +  "train/*/*") 
    test = glob(data_path + "test/*/*") 
    
    # Count max number of LRs in all scenes
    max_lrs = 0
    for scene in glob(data_path + "train/*/*") + glob(data_path + "test/*/*"):
        lrs = len(glob(scene + "/LR*"))
        if max_lrs < lrs:
            max_lrs = lrs
            
    for scene in train + test:
        save_scene(scene)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate TFRecords from the ProbaV Data")
    parser.add_argument("--data_dir", help="path to the directory containing the data", type=str, default="Data/")
    args = parser.parse_args()
    main(args.data_dir)