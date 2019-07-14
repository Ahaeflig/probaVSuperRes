import argparse

import tensorflow as tf
from glob import glob
import os
import os.path

from tensorflow import keras

import pickle

import skimage
from skimage import io 

# Import data loader
from data_loader import MultipleDataLoader

# Import model
from SRCNN import SRCNN

# Function for model training
from training_helpers import compute_loss, train_step, show_pred, save_pred

""" Program to train a network for Super Resolution

"""


def main(epochs: int, learning_rate: float, batch_size: int, save_interval: int, model_path: str, verbose: bool):
    
    data_dir = "DataNormalized/"
    DataLoader = MultipleDataLoader(data_dir)

    train_dataset = tf.data.TFRecordDataset(glob(data_dir +  "train/*/*/multiple.tfrecords"))
    train_dataset = train_dataset.shuffle(len(glob(data_dir +  "train/*/*/multiple.tfrecords")))
    train_dataset = train_dataset.map(lambda x: DataLoader.parse_multiple_fixed(x, augment=True), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    train_dataset = train_dataset.batch(batch_size)

    # Create a Model folder if it doesn't exist
    os.makedirs("Model/", exist_ok=True)
    
    # We load a pre-exisiting model if it exists
    if os.path.isfile(model_path): 
        if verbose:
            print("Loading model from " + model_path)
        
        model = keras.models.load_model(model_path, compile=False)
        # I encode the current epoch in the model file name: modelName_epoch.h5
        current_epoch = int(model_path.split("_")[1].split(".")[0])

    else:
        if verbose:
            print("Creating new model, will save at Model/residual.h5 after training")
        model_path = "Model/residual_0.h5"
        current_epoch = 0
        model = SRCNN().model
        
    model_base_path = model_path.split("_")[0]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    
    if verbose:
        print("Model Training starting")
    
    # Training
    train_losses = []
    for epoch in range(current_epoch, current_epoch + epochs):
        
        '''
        Temporarily predict model output because model.save appears to be broken TODO
        '''
        if (epoch + 1) % 30 == 0:
            if verbose:
                print("Predicting output")
            predict_output(model, epoch)
        
        loss = 0
        for lrs, hr in train_dataset:
            loss += train_step(lrs, hr, model, optimizer)
        train_losses.append(loss)
        
        # Save the model TODO appears to be broken, chained save problem?
        if (epoch + 1) % save_interval == 0:
            if verbose:
                print("Saving model")
            model.save(model_base_path + "_" + str(epoch) + ".h5", include_optimizer=False)

        # Save a predicted sample
        if (epoch + 1) % 30 == 0:
            for lrs, hr in train_dataset.take(1):
                save_pred(model, lrs, hr, "ResidualCNN", epoch)
        
        if verbose:
            print('epoch ' + str(epoch) + ' current train loss: ' + str(loss))

            
    model.save(model_base_path + "_" + str(epoch) + ".h5", include_optimizer=False)
    
    # could convert to tensorboard logging if time
    with open('residual_losses.pickle', 'wb') as handle:
        pickle.dump(train_losses, handle)
        
    # Print current Adam state for next training, could save somehow
    print(optimizer.get_config())
    

"""
***Temporarily*** predict model output because model.save appears to be broken TODO
"""
def predict_output(model, epoch):
    test_scenes = glob("DataNormalized/test/*/*/multiple.tfrecords")
    DataLoader = MultipleDataLoader("DataNormalized/")
    
    # We load test data from TFRecords because LRs are already normalized
    test_dataset = tf.data.TFRecordDataset(test_scenes)
    test_dataset = test_dataset.map(lambda x: DataLoader.parse_multiple_fixed(x, augment=False)) 
    test_dataset = test_dataset.batch(1)
    
    output_files = [record_path.split("/")[-2] + ".png" for record_path in test_scenes]
    
    save_path = "Result/Predicted/SRCNN_" + str(epoch) + "/"
    os.makedirs(save_path, exist_ok=True)
    
    for lr_hr, file_name in zip(test_dataset, output_files):
        lrs, _ = lr_hr
        output = model(lrs)
        output = skimage.img_as_uint(output[0])
        
        io.imsave(save_path + file_name, output)
    

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResidualCNN with multiple inputs")
    parser.add_argument("-e", "--epoch", help="Number of epochs to train", type=int, default=1000)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", help="Number of scene per batch", type=int, default=4)
    parser.add_argument("--save_interval", help="Save model every epoch", type=int, default=20)
    parser.add_argument("--model_path", help="modelName_epoch.h5 path (for loading/saving)", type=str, default="residual")
    
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")

    args = parser.parse_args()
    
    main(args.epoch, args.learning_rate, args.batch_size, args.save_interval, args.model_path, args.verbose)