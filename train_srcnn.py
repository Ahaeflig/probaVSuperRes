import argparse

import tensorflow as tf
from glob import glob
import os
import os.path

from tensorflow import keras

import skimage
from skimage import io 

# Import data loader
from data_loader import MultipleDataLoader

# Import model
from SRCNN import SRCNN

# Function for model training
from training_helpers import clearMSE_metric, compute_loss, show_pred, save_pred


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
    
    # TODO make these params
    lr_channels = 5
    data_dir = "DataNormalized/"
    
    DataLoader = MultipleDataLoader(data_dir)
    
    train_files = glob(data_dir +  "train/*/*/multiple.tfrecords")
    
    train_dataset = tf.data.TFRecordDataset(train_files)
    # Map each file to the parsing funciton, enabling data augmentation
    train_dataset = train_dataset.map(lambda x: DataLoader.parse_multiple_fixed(x, augment=True, num_lrs = lr_channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # reshuffle_each_iteration works only when combined with repeat
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.shuffle(len(train_files))

    # Set the batch size
    train_dataset = train_dataset.batch(batch_size)

    # Create a Model folder if it doesn't exist
    os.makedirs("Model/", exist_ok=True)
    
    # We load a pre-exisiting model if it exists
    if os.path.isfile(model_path): 
        if verbose:
            print("Loading model from " + model_path)
        
        custom_object = {'compute_loss': compute_loss, 'clearMSE_metric': clearMSE_metric}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_object)
        
    else:
        if verbose:
            print("Creating new model, will save at Model/residual.h5 after training")
            
        model = SRCNN(channel_dim=lr_channels, include_batch_norm = False).model
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        filepath = "Model/SRCNN/{epoch:02d}.hdf5"
        os.makedirs("Model/SRCNN/", exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='train_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
        
        model.compile(optimizer, compute_loss, metrics=[clearMSE_metric])

        callbacks_list = [checkpoint]
        
    if verbose:
        print("Model Training starting:")
    
    model.fit_generator(train_dataset, steps_per_epoch=len(train_files) / batch_size, epochs=epochs, use_multiprocessing=True, callbacks=callbacks_list)
    
    if verbose:
        print("Model Training finished:")
    
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