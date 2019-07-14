import argparse

import tensorflow as tf
from glob import glob
import os
import os.path

import skimage
from skimage import io 

from tensorflow import keras


def get_output_name(tf_record_path):
    scene_name = tf_record_path.split("/")[-2]
    return scene_name + ".png"
    

def main(model_path: str, output_dir: str):
    
    model = keras.models.load_model(model_path)
    
    test_scenes = glob(data_dir +  "test/*/*/multiple.tfrecords")
    
    # We load test data from TFRecords because LRs are already normalized
    test_dataset = tf.data.TFRecordDataset(test_scenes)
    test_dataset = test_dataset.map(lambda x: DataLoader.parse_multiple_fixed(x, augment=False)) 
    test_dataset = test_dataset.batch(1)
    
    # Same order as the data iterator above
    output_files = [get_output_name(record_path) +  for record_path for fname in test_scenes]
    
    for lrs, hr, file_name in zip(test_dataset, output_names):
        output = model(test_dataset)
        output = skimage.img_as_uint(output)
        io.imsave(imgpath, output_dir + file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResidualCNN with multiple inputs")
    parser.add_argument("model_path", help="path to model (for loading)", type=str)
    parser.add_argument("--output_dir", help="path to save predictions", type=str, default="Result/Predicted/")

    args = parser.parse_args()
    
    main(args.model_path, args.output_dir)