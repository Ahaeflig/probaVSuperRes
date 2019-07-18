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


def main(epochs: int, learning_rate: float, batch_size: int, save_interval: int, model_path: str, verbose: bool):
    """ Program to train a SRCNN for Super Resolution

        Args:
            epochs: the number of epochs to train the model on
            learning_rate: the optimizer's initial learning rate
            batch_size: how many samples are run in parallel during forward/backward pass
            save_interval: when to save model 
            model_path: path to the model
            verbose: flag to enable or disable print outputs
    """
    
    data_dir = "DataNormalized/"
    DataLoader = MultipleDataLoader(data_dir)

    train_files = glob(data_dir +  "train/*/*/multiple.tfrecords")
    
    train_dataset = tf.data.TFRecordDataset(train_files)
    # reshuffle_each_iteration works only for the repeat operation
    train_dataset = train_dataset.shuffle(len(train_files), reshuffle_each_iteration=True)
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
        print("Model Training starting: " )
        #print(len(train_dataset) )
    
    # Training
    train_losses = []
    for epoch in range(current_epoch, current_epoch + epochs):
        
        loss = 0
        for lrs, hr in train_dataset:
            loss += train_step(lrs, hr, model, optimizer)
        
        train_losses.append(loss)
        
        if (epoch + 1) % save_interval == 0:
            if verbose:
                print("Saving model")
            model.save(model_base_path + "_" + str(epoch) + ".h5", include_optimizer=False)

        # Save a predicted sample
        if (epoch + 1) % save_interval == 0:
            for lrs, hr in train_dataset.take(1):
                save_pred(model, lrs, hr, "ResidualCNN", epoch)
        
        if verbose:
            print('epoch ' + str(epoch) + ' current train loss: ' + str(loss))
            
        # Reshuffle dataset, see https://github.com/tensorflow/tensorflow/issues/27680
        train_dataset = train_dataset.shuffle(len(train_files))
            
    model.save(model_base_path + "_" + str(epoch) + ".h5", include_optimizer=False)
    
    # could convert to tensorboard logging if time
    with open('residual_losses.pickle', 'wb') as handle:
        pickle.dump(train_losses, handle)
        
    # Print current Adam state for next training, could save somehow
    print(optimizer.get_config())
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResidualCNN with multiple inputs")
    parser.add_argument("-e", "--epoch", help="Number of epochs to train", type=int, default=41)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", help="Number of scene per batch", type=int, default=4)
    parser.add_argument("--save_interval", help="Save model every epoch", type=int, default=20)
    parser.add_argument("--model_path", help="modelName_epoch.h5 path (for loading/saving)", type=str, default="residual")
    
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")

    args = parser.parse_args()
    
    main(args.epoch, args.learning_rate, args.batch_size, args.save_interval, args.model_path, args.verbose)