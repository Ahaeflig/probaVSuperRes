import argparse

import tensorflow as tf
from glob import glob
import os
import os.path

from tensorflow import keras

import time
import pickle

# Import data loader
from MultipleDataLoader import MultipleDataLoader

# Import model
from MultipleResidualCNN import MultipleResidualCNN

# Function for model training
from TrainingHelpers import compute_loss, train_step, show_pred, save_pred

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
        model = MultipleResidualCNN().model
        
    model_base_path = model_path.split("_")[0]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    
    if verbose:
        print("Model Training starting")
    
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

        if (epoch + 1) % 25 == 0:
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