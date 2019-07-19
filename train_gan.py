import argparse

import tensorflow as tf
from glob import glob
import os
import os.path

from tensorflow import keras

# Import data loader
from data_loader import MultipleDataLoader

# Import models
from SRGAN import SRGAN
from SRCNN import SRCNN

# Function for model training
from training_helpers import discriminator_loss, compute_loss, clearMSE_metric, train_step_gan


def main(epochs: int, learning_rate: float, batch_size: int, save_interval: int, model_path: str, verbose: bool):
    """ Program to train an SRGAN model

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

    train_dataset = tf.data.TFRecordDataset(train_files)
    # Map each file to the parsing funciton, enabling data augmentation
    train_dataset = train_dataset.map(lambda x: DataLoader.parse_multiple_fixed(x, augment=True, num_lrs = lr_channels), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # reshuffle_each_iteration works only when combined with repeat
    train_dataset = train_dataset.shuffle(len(train_files))
    
    # Set the batch size
    train_dataset = train_dataset.batch(batch_size)

    # Create a Model folder if it doesn't exist
    os.makedirs("Model/", exist_ok=True)
    
    # We load a pre-exisiting model if it exists
    if os.path.isfile(model_path): 
        raise NotImplementedError("Loading SRGAN model not implemented yet (optimizer state not saved for now)")
        
        if verbose:
            print("Loading model from " + model_path)
        
        print("Model loading not  implemented yet")
        model = keras.models.load_model(model_path)
        # I encode the current epoch in the model file name: modelName_epoch.h5
        current_epoch = int(model_path.split("_")[1].split(".")[0])

    else:
        if verbose:
            print("Creating new model, will save at Model/srgan_epoch.h5 after training")
           
        generator = SRCNN(channel_dim = lr_channels, number_residual_block = 9, include_batch_norm = False).model
        srgan = SRGAN(generator, channel_dim=lr_channels, include_batch_norm = False)
        model = srgan.model
        generator = srgan.generator
        discriminator = srgan.discriminator

        optimizer_gen = tf.keras.optimizers.Adam(0.0001)
        optimizer_discr = tf.keras.optimizers.Adam(0.0001)
        
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer =  tf.keras.optimizers.Adam(learning_rate)
    
    if verbose:
        print("SRGAN Model Training starting for " + str(epochs) + " epochs")
        
    # Controls if we should update the generator or not
    gen_train = True

    dataset_epoch = train_dataset.shuffle(len(train_files))
    
    # Training
    train_gen_losses = []
    train_discr_losses = []
    for epoch in range(current_epoch, current_epoch + epochs):
        
        disc_loss = 0
        gen_loss = 0
        for lrs, hr in dataset_epoch:
            gl, dl = train_step_gan(lrs, hr, generator, discriminator, optimizer_gen, optimizer_discr, gen_train)
            gen_loss += gl
            disc_loss += dl

        disc_losses.append(disc_loss)
        gen_losses.append(gen_loss)
        
        if (epoch + 1) % save_interval == 0:
            if verbose:
                print("Saving model")
            model.save("Model/SRGAN/" + str(epoch) + ".hdf5")

        if (epoch + 1) % 25 == 0:
            for lrs, hr in train_dataset.take(1):
                save_pred("Model/SRGAN/" + "SRGAN", epoch)

        if verbose:
            print('epoch ' + str(epoch) + ' current losses: (gen / disc) ' + str(gen_loss.numpy()) + "/" + str(discr_loss.numpy()))
            
        # see https://github.com/tensorflow/tensorflow/issues/27680
        dataset_epoch = train_dataset.shuffle(len(train_files))
        
        if verbose:
            print("Model Training finished:")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SRGAN with multiple (LRs) inputs")
    parser.add_argument("-e", "--epoch", help="Number of epochs to train", type=int, default=1000)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", help="Number of scene per batch", type=int, default=4)
    parser.add_argument("--save_interval", help="Save model every epoch", type=int, default=1)
    parser.add_argument("--model_path", help="modelName_epoch.h5 path (for loading/saving)", type=str, default="residual")
    
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")

    args = parser.parse_args()
    
    main(args.epoch, args.learning_rate, args.batch_size, args.save_interval, args.model_path, args.verbose)