import argparse

import tensorflow as tf
from glob import glob
import os
import os.path

from tensorflow import keras

from data_loader import MultipleDataLoader

from supreshelper import *


BASELINE = "baseline"

def main(data_path: str, model_path: str):
    """ Computes the baseline cSPVR score on the train set or on a trained model if model_path is specified

    """
    
    # TODO make this portion reusable, data loader could handle TFRecordDataset itself
    DataLoader = MultipleDataLoader(data_path)
    train_set = glob(data_path +  "train/*/*/")

    train_dataset = tf.data.TFRecordDataset(train_set + "multiple.tfrecords")
    train_dataset = train_dataset.map(lambda x: DataLoader.parse_multiple_fixed(x, augment=True), num_parallel_calls=tf.data.experimental.AUTOTUNE) 

    if model_path == BASELINE:
        print("Super resolving with baseline")
        srs = [baseline_upscale(scene) for scene in train_set]

    else:
        print("Super resolving with " + model_path)
        model = keras.models.load_model(model_path, compile=False)
        srs = [model(lr) for lr,_ in train_dataset]
        pass
    
    scores = score_images(srs, train_set)
    print("score: " + np.mean(scores))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResidualCNN with multiple inputs")
    parser.add_argument("--data_path", help="(Normalized if using SRCNN)Data set base folder path", type=str, default="Data/")
    parser.add_argument("--model_path", help="modelName_epoch.h5: path to model ", type=str, default=BASELINE)

    args = parser.parse_args()


    main(args.data_path, args.model_path)
