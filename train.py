import argparse

import tensorflow as tf
from tensorflow import keras

import numpy as np
from typing import Dict, List

# Import data loader
from data_loader import MultipleDataLoader

# Import models
from SRGAN import SRGAN
from SRCNN import SRCNN
from SR_UNET import SR_UNET

from enum import Enum

import os

def verbose_print(sentence: str, verbose: bool):
    if verbose:
        print(sentence)
        
        
class Models(Enum):
    SRCNN = 0
    SRGAN = 1


class Losses():
    """ Collection of losses used throughout the project and utils to compute the loss
    """
    
    
    @staticmethod
    def cMSE(hr_masked, generated_masked):
        """As defined in https://kelvins.esa.int/proba-v-super-resolution/scoring/
        MSE loss that is not sensitive to average difference

        Args:
            hr_masked: the masked HR image
            generated_masked: the masked generated image

        Return:
            The mean squared error between both image with a corrected bias

        """

        bias = tf.math.reduce_mean(hr_masked - generated_masked) 
        loss = tf.math.reduce_mean(tf.pow(hr_masked - (generated_masked + bias), 2))
        return loss
    
    
    @staticmethod
    def cMAE(hr_masked, generated_masked):
        """ MAE loss in the same vein as clearMSE

        Args:
            hr_masked: the masked HR image
            generated_masked: the masked generated image

        Return:
            The mean average error between both image with a corrected bias
        """

        bias = tf.math.reduce_mean(hr_masked - generated_masked) 
        loss = tf.math.reduce_mean(tf.abs(hr_masked - (generated_masked + bias)))
        return loss
    
    
    @staticmethod
    def log10(x):
        """ Compute base 10 log
        """
        numerator = tf.math.log(x)
        # tf.math.log(tf.constant(10.0))
        denominator = 2.3025851
        return numerator / denominator

    
    @staticmethod
    def cPSNR(hr, sr):
        """ clear Peak Signal to Noise Ratio loss as defined here:
            https://kelvins.esa.int/proba-v-super-resolution/scoring/
        """
        # 1e-10: makes it such that if the cMSE is 0, then we get the best score defined as 100
        return -10.0 * Losses.log10(Losses.cMSE(hr, sr) + 1e-10)
    
    
    @staticmethod
    def cPSNR_metric(hr, sr):
        """ Apply the mask itself and calls cPSNR
        """
        
        hr_, sr_= Losses.apply_mask(hr, sr)
        return Losses.cPSNR(hr_, sr_)
        
        
    @staticmethod
    def SSIM(hr, sr):
        # TODO what would be nice for sigma?
        diff_loss = 1.0 - tf.image.ssim(hr, sr, 1.0, filter_size=3, filter_sigma=0.9, k1=0.01, k2=0.03)
        return tf.math.reduce_mean(diff_loss)
    
    
    @staticmethod
    def MS_SSIM(hr, sr):
        # Multiscale SSIM
        diff_loss = 1.0 - tf.image.ssim_multiscale(hr, sr, 1.0, filter_size=3, filter_sigma=1.2, k1=0.01, k2=0.03)
        return tf.math.reduce_mean(diff_loss)
    
    
    @staticmethod
    def apply_mask(hr, sr):
        """ Finds np.NaN values in the HR image and set those pixel to 0 (False) in hr and generated.

            Args:
                hr: High resolution image where obstructed pixel are encoded as nan values
                sr: generated image by the model which

            Return:
                The modified (masked) HR and generated images

        """
        hr_ = tf.where(tf.math.is_nan(hr), 0.0, hr)
        sr_ = tf.where(tf.math.is_nan(hr), 0.0, sr)

        return hr_, sr_
    
    
    
'''
==============================
            CNN
==============================
'''
    
class SRCNNTrainer:
    """ Class that handles training of the SRCNN Model.
    """
    
    def __init__(self, training_params: Dict, srcnn_model_parameters: Dict):
        
        # Hyperparameters
        self.losses = training_params["losses"]
        self.losses_weights = training_params["losses_weights"]
        self.epochs = training_params["epochs"]
        self.optimizer = training_params["optimizer"]
        self.verbose = training_params["verbose"]
        
        self.model_path = ""
        if "model_path" in training_params:
            self.model_path = training_params["model_path"]
        
        # Model parameters
        self.channel_dim = srcnn_model_parameters["channel_dim"]
        self.number_residual_block = srcnn_model_parameters["number_residual_block"]
        self.batch_norm = srcnn_model_parameters["batch_norm"]
        
        self.model = self.build_model()
        
        
    def build_model(self):
        if os.path.isfile(self.model_path):
            verbose_print("Loading model from " + self.model_path, self.verbose)
            custom_object = {'custom_loss': self.custom_loss, 'cPSNR_metric': Losses.cPSNR_metric}
            model = tf.keras.models.load_model(self.model_path, custom_objects=custom_object)
            
        else:
            verbose_print("Creating new model, saving at \"Model/SRCNN/\" ", self.verbose)
            model = SRCNN(channel_dim=self.channel_dim, number_residual_block = self.number_residual_block, include_batch_norm = self.batch_norm).model 
            model.compile(self.optimizer, self.custom_loss, metrics=[Losses.cPSNR_metric])
        
        return model
    
    
    @tf.function
    def custom_loss(self, hr, sr):
        hr_, sr_ = Losses.apply_mask(hr, sr)
        
        loss = 0 
        for loss_func, weight in zip(self.losses, self.losses_weights):
            loss += tf.math.multiply(weight, loss_func(hr_, sr_))
            
        return loss
        
    
    def fit(self, data_loader):
        """
            Args:
            data_loader: a MultipleDataLoader instance.
        """
        verbose_print("Starting model training", self.verbose)
        
        # Save callback
        filepath = "Model/SRCNN/{epoch:02d}.hdf5"
        os.makedirs("Model/SRCNN/", exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='train_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
            
        # Reduce LR callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2, cooldown=1, verbose=self.verbose)
        
        callbacks_list = [checkpoint, reduce_lr]
        
        self.model.fit_generator(data_loader(), steps_per_epoch= len(data_loader.files) / data_loader.batch_size, epochs=self.epochs, use_multiprocessing=True, callbacks=callbacks_list)
        
        verbose_print("Model training finished", self.verbose)
    
    

    
'''
==============================
           GAN
==============================
'''
    
class SRGANTrainer:
    """ Class that handles training of the SRGAN Model.
    """
    
    def __init__(self, training_params: Dict, srcnn_model_parameters: Dict):
        
        # Hyperparameters
        self.losses = training_params["losses"]
        self.losses_weights = training_params["losses_weights"]
        self.epochs = training_params["epochs"]
        self.optimizer_gen = training_params["optimizer_gen"]
        self.optimizer_discr = training_params["optimizer_discr"]
        self.verbose = training_params["verbose"]
        self.pretrained_generator_path = training_params["args.pretrained_generator_path"]
        
        self.model_path = ""
        if "model_path" in training_params:
            self.model_path = training_params["model_path"]
        
        # Model parameters
        self.channel_dim = srcnn_model_parameters["channel_dim"]
        self.number_residual_block = srcnn_model_parameters["number_residual_block"]
        self.batch_norm = srcnn_model_parameters["batch_norm"]
        
        self.generator = None
        self.discriminator = None
        self.model = self.build_model()
        
    def build_model(self):
        if os.path.isfile(self.model_path):
            raise NotImplementedError("Retraining SRGAN model not implemented yet")
            
        else:
            verbose_print("Creating new model, saving at \"Model/SRGAN/\" ", self.verbose)
            
            generator = SRCNN(channel_dim=self.channel_dim, number_residual_block = self.number_residual_block, include_batch_norm = self.batch_norm).model 
            
            if (os.path.isfile(self.pretrained_generator_path)):
                verbose_print("Loading pre-trained generator from " + self.pretrained_generator_path, self.verbose)
                custom_object = {'custom_loss': SRCNNTrainer.custom_loss, 'cPSNR_metric': Losses.cPSNR_metric}
                gen_weights = keras.models.load_model(self.pretrained_generator_path, custom_objects=custom_object).weights
                generator.load_weights(gen_weights)

            srgan = SRGAN(generator, channel_dim=self.channel_dim, include_batch_norm = self.batch_norm)
            self.generator = srgan.generator
            self.discriminator = srgan.discriminator

        return srgan.model
    
    
    @tf.function
    def custom_loss(self, hr, sr):
        #hr_, sr_ = Losses.apply_mask(hr, sr)
        
        loss = 0 
        for loss_func, weight in zip(self.losses, self.losses_weights):
            loss += tf.math.multiply(weight, loss_func(hr, sr))
            
        return loss
    
    
    def fit(self, data_loader, current_epoch=0):
        """ Trains the model on the dataset
        
            Args:
            data_loader: a MultipleDataLoader instance.
        """
        
        verbose_print("Starting SRGAN model training", self.verbose)
        
        # Save callback
        os.makedirs("Model/SRGAN/", exist_ok=True)
            
        # Reduce LR callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.00002, patience=2, cooldown=2, verbose=self.verbose)
        
        callbacks_list = [reduce_lr]
        
        # Controls if we should update the generator or not
        gen_train = True
        
        for epoch in range(current_epoch, current_epoch + self.epochs):
            
            disc_loss = 0.0
            gen_loss = 0.0
            cPSNR_loss = 0.0
            
            for lrs, hr in data_loader.get_shuffled_copy():
                
                gl, dl, psnr = self.train_step_gan(lrs, hr, self.generator, self.discriminator, self.optimizer_gen, self.optimizer_discr, data_loader.batch_size, gen_train)
                
                gen_loss += gl
                disc_loss += dl
                cPSNR_loss += psnr
                
            disc_loss = tf.math.reduce_mean(disc_loss)
            gen_loss = tf.math.reduce_mean(gen_loss)
            cPSNR_loss = tf.math.reduce_mean(cPSNR_loss)
                
            if (epoch + 1) % 1 == 0:
                verbose_print("Saving model", self.verbose)
                self.model.save("Model/SRGAN/" + str(epoch) + ".hdf5")
                
            # TODO implement LR on plateau
            if epoch > 80:
                # Learning rate decays:
                # lr * lr_decay_target * (epoch / max_epoch)
                self.optimizer_gen.lr = self.optimizer_gen.lr * 0.08 ** ((epoch + 80) / (self.epochs + 80) )
                self.optimizer_discr.lr = self.optimizer_discr.lr * 0.08 ** ((epoch + 80) / (self.epochs + 80))
            
            
            loss_report = 'epoch: ' + str(epoch) + ' current losses: (gen / disc) ' + str(gen_loss.numpy()) + " / " + str(disc_loss.numpy()) + " cPSNR: " + str(cPSNR.numpy())
                                                                             
            verbose_print(loss_report, self.verbose)
            
        verbose_print("Model training finished", self.verbose)
    
    
    def discriminator_loss(self, disc_hr_output, disc_sr_output, batch_size):
        real_loss = (tf.ones_like(disc_hr_output) - tf.random.uniform([batch_size, 1]) * 0.1) - disc_hr_output
        real_loss = tf.clip_by_value(real_loss, 0.0, 1.0)

        generated_loss = tf.random.uniform([batch_size, 1]) * 0.1 + disc_sr_output

        return 0.5 * tf.math.reduce_mean((real_loss + generated_loss))
    
    
    def generator_loss(self, disc_sr, hr, sr, batch_size, lambda_ = 100):
        # we try to trick the discriminator to predict our generated image to be considered as valid ([1])
        gan_loss = tf.math.reduce_mean((tf.ones_like(disc_sr) - tf.random.uniform([batch_size, 1]) * 0.1) - disc_sr)
        gan_loss = tf.clip_by_value(gan_loss, 0.0, 1.0)

        # We also want the image to look as similar as possible to the HR images
        img_loss = self.custom_loss(hr, sr)

        # Lambda weights how much we value each loss
        return gan_loss + (lambda_ * img_loss)
    
    
    @tf.function
    def train_step_gan(self, lrs, hr, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, gen_train = False):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # Generate one batch
            sr = self.generator(lrs)
            
            # Change NaN to 0s in both images
            hr_, sr_ = Losses.apply_mask(hr, sr)
            
            metric = Losses.cPSNR(hr_, sr_)

            # Compute discriminator predictions on real and generated images
            disc_hr = discriminator(hr_, training=True)
            disc_sr = discriminator(sr_, training=True)

            # Computes losses
            disc_loss = self.discriminator_loss(disc_hr, disc_sr, batch_size)
            gen_loss = self.generator_loss(disc_sr, hr_, sr_, batch_size)

            # We compute gradients
            discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            
            # We can optionally freeze the gen update
            if tf.equal(gen_train, True):
                pass
                #generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                #generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            
            # Update weights
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        return gen_loss, disc_loss, metric
    
    
    
'''
==============================
            U_NET
==============================
'''
    
    
class SR_UNETTrainer:
    """ Class that handles training of the SR_UNET Model.
    """
    
    def __init__(self, training_params: Dict, sr_unet_model_parameters: Dict):
        
        # Hyperparameters
        self.losses = training_params["losses"]
        self.losses_weights = training_params["losses_weights"]
        self.epochs = training_params["epochs"]
        self.optimizer = training_params["optimizer"]
        self.verbose = training_params["verbose"]
        
        self.model_path = ""
        if "model_path" in training_params:
            self.model_path = training_params["model_path"]
        
        # Model parameters
        self.channel_dim = sr_unet_model_parameters["channel_dim"]
        self.batch_norm = sr_unet_model_parameters["batch_norm"]
        
        self.model = self.build_model()
        
        
    def build_model(self):
        if os.path.isfile(self.model_path):
            verbose_print("Loading model from " + self.model_path, self.verbose)
            custom_object = {'custom_loss': self.custom_loss, 'cPSNR_metric': Losses.cPSNR_metric}
            model = tf.keras.models.load_model(self.model_path, custom_objects=custom_object)
            
        else:
            verbose_print("Creating new model, saving at \"Model/SR_UNET/\" ", self.verbose)
            model = SRCNN(channel_dim=self.channel_dim, include_batch_norm = self.batch_norm).model 
            model.compile(self.optimizer, self.custom_loss, metrics=[Losses.cPSNR_metric])
        
        return model
    
    
    @tf.function
    def custom_loss(self, hr, sr):
        hr_, sr_ = Losses.apply_mask(hr, sr)
        
        loss = 0 
        for loss_func, weight in zip(self.losses, self.losses_weights):
            loss += tf.math.multiply(weight, loss_func(hr_, sr_))
            
        return loss
        
    
    def fit(self, data_loader):
        """
            Args:
            data_loader: a MultipleDataLoader instance.
        """
        verbose_print("Starting model training", self.verbose)
        
        # Save callback
        filepath = "Model/SR_UNET/{epoch:02d}.hdf5"
        os.makedirs("Model/SR_UNET/", exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='train_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
            
        # Reduce LR callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2, cooldown=1, verbose=self.verbose)
        
        callbacks_list = [checkpoint, reduce_lr]
        
        self.model.fit_generator(data_loader(), steps_per_epoch= len(data_loader.files) / data_loader.batch_size, epochs=self.epochs, use_multiprocessing=True, callbacks=callbacks_list)
        
        verbose_print("Model training finished", self.verbose)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Super resolution models")
    parser.add_argument("--data_path", help="Number of epochs to train", type=str, default="DataTFRecords/train/")
    parser.add_argument("--model", help="Which type of model: gan | cnn", type=str, default="gan")
    
    parser.add_argument("-e", "--epoch", help="Number of epochs to train", type=int, default=100)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", help="Number of scene per batch", type=int, default=4)
    
    parser.add_argument("--augment", help="Turn on data augmentation", action="store_true")
    
    parser.add_argument("--num_channel", help="Number of dimension of the input (LRs)", type=int, default=5)
    
    parser.add_argument("--model_path", help="modelName_epoch.h5 path (for loading/saving), cnn only", type=str, default="Model/SRCNN/20.hd5f")
    
    parser.add_argument("--pretrained_generator_path", help="modelName_epoch.h5 pre-trained generator path", type=str, default="Model/SRCNN/20.hd5f")
    
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")

    args = parser.parse_args()
    
    generator_model_parameters = {
        'channel_dim': args.num_channel,
        'number_residual_block': 3,
        'batch_norm': False,
    }
    

    if model == "gan" or model == "srgan":
        loader = MultipleDataLoader("DataTFRecords/train/", args.batch_size, False, args.augment, args.num_channel)
        
        srgan_training_hyperparams = {
            'losses': [Losses.cMSE, Losses.MS_SSIM],
            'losses_weights': [0.8, 0.2],
            'epochs': args.epoch,
            'optimizer_gen': tf.keras.optimizers.Adam(args.learning_rate),
            'optimizer_discr': tf.keras.optimizers.Adam(args.learning_rate),
            'verbose': args.verbose,
            'pretrained_generator_path': args.pretrained_generator_path
        }
        
        trainer = SRGANTrainer(srgan_training_hyperparams, generator_model_parameters)
        
    elif model == "cnn":
        loader = MultipleDataLoader("DataTFRecords/train/", args.batch_size, True, args.augment, args.num_channel)
        
        srcnn_training_hyperparams = {
            'losses': [Losses.cMSE, Losses.MS_SSIM],
            'losses_weights': [5.0, 1.0],
            'epochs': args.epoch,
            'optimizer': tf.keras.optimizers.Adam(args.learning_rate),
            'verbose': args.verbose,
            'model_path': args.model_path
        }
        
        trainer = SRCNNTrainer(generator_model_parameters, srcnn_model_parameters)
        
    trainer.fit(loader)
    
    

'''
==============================
        Visualization
==============================
'''

import matplotlib.pyplot as plt
    
def predict_and_show(model, lrs, hr, max_lr=5):
    predicted = model.predict(lrs)
    plt.figure(figsize=(15,15))
    
    display_list = [lrs[0][:,:,0], predicted[0][:,:,0], hr[0][:,:,0]]
    title = ['Input Image (1 of ' + str(max_lr) + ')', 'Predicted Image', 'Real image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
