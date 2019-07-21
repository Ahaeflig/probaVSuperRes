import argparse

import tensorflow as tf
from glob import glob
import os
import os.path

from tensorflow import keras

from data_loader import MultipleDataLoader

from train import SRCNNTrainer, SRGANTrainer, Losses

from supreshelper import *


BASELINE = "baseline"

def main(data_path: str, model_path: str, num_channel: int, gan: bool):
    """ Computes the baseline cSPVR score on the train set or using a trained model if model_path is specified

    """
    
    # Disable tf logs and warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    data_load = MultipleDataLoader(data_path, 1, False, False, False, num_channel)
    train_set = glob(data_path + "*/*/")
    
    assert(len(train_set) > 1), "No scenes found, check out path to data"
    
    if model_path == BASELINE:
        print("Super resolving with baseline")
        srs = [baseline_upscale(scene) for scene in train_set]
        
    else:
        print("Super resolving with " + model_path)
        custom_object = {'custom_loss': None, 'cPSNR_metric': Losses.cPSNR_metric}
        model = keras.models.load_model(model_path, custom_objects=custom_object, compile=False)
        
        if gan:  
            # need to take generator output only
            srs = [model(lrs)[0][0][:,:,0].numpy() for lrs, _ in data_load()]
        else:
             srs = [model(lrs)[0][:,:,0].numpy() for lrs, _ in data_load()]
    
    scores = []
    i = 0
    for sr in srs:
        scores.append(score_image_fast(sr, train_set[i]))
        i += 1  
        
    print("score: " + str(np.mean(scores)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResidualCNN with multiple inputs")
    parser.add_argument("--data_path", help="(Normalized if using SRCNN)Data set base folder path", type=str, default="DataTFRecords/train/")
    parser.add_argument("--model_path", help="modelName_epoch.h5: path to model ", type=str, default=BASELINE)
    parser.add_argument("--num_channel", help="Number of dimension of the input (LRs)", type=int, default=5)
    parser.add_argument("--gan", help="Turn on GAN moode (predict with generator)", action="store_true")
    

    args = parser.parse_args()

    main(args.data_path, args.model_path, args.num_channel, args.gan)
