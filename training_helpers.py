import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

import os

"""
Helper file with functions written to perform the optimization of a neural net.
"""


'''
==============================
        CNN Helpers
==============================
'''

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


def clearMSE(hr_masked, generated_masked):
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


def clearMSE_metric(hr, sr):
    hr = apply_mask(hr, sr)
    bias = tf.math.reduce_mean(hr - sr) 
    loss = tf.math.reduce_mean(tf.pow(hr - (sr + bias), 2))
    return loss
    

def clearMAE(hr_masked, generated_masked):
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


def similarity_loss(hr, sr):
    # TODO what would be nice for sigma?
    diff_loss = 1.0 - tf.image.ssim(hr, sr, 1.0, filter_size=3, filter_sigma=1.2, k1=0.01, k2=0.03)
    return tf.math.reduce_mean(diff_loss)


def compute_loss(hr, sr):
    hr_, sr_ = apply_mask(hr, sr) 
    return similarity_loss(hr_, sr_)


@tf.function
def predict_and_compute_loss(lrs, hr, model, training=True):
    sr = model(lrs, training)
    hr_masked, sr_masked = applyMask(hr, sr) 
    return clearMSE(hr_masked, sr_masked) + clearMAE(hr_masked, sr_masked)

@tf.function
def train_step(lrs, hr, model, optimizer):
    with tf.GradientTape() as model_tape:
        loss = predict_and_compute_loss(lrs, hr, model)
        # Compute gradient for parameter
        model_gardients = model_tape.gradient(loss, model.trainable_variables)
        # Apply the gradient to the variables, model weights are updated here
        optimizer.apply_gradients(zip(model_gardients, model.trainable_variables))
        
    return loss


'''
==============================
        GAN Helpers
==============================
'''

def discriminator_loss(disc_hr_output, disc_sr_output):
    # real_loss = 1.0 - disc_real_output
    
    real_loss = (tf.ones_like(disc_hr_output) - tf.random.uniform(disc_hr_output.shape) * 0.1) - disc_hr_output
    real_loss = tf.clip_by_value(real_loss, 0.0, 1.0)
    
    # generated_loss =  tf.math.abs(disc_generated_output - 1.0)
    # generated_loss =  disc_generated_output
    
    generated_loss = tf.random.uniform(disc_sr_output.shape) * 0.1 + disc_sr_output
    
    return 0.5 * tf.math.reduce_mean((real_loss + generated_loss))
    
    # total_disc_loss = 0.5 * (real_loss + generated_loss)
    # return tf.math.reduce_mean(total_disc_loss)

def generator_loss(disc_sr, hr, sr, lambda_ = 100):
    # we try to trick the discriminator to predict our generated image to be considered as valid ([1])
    
    #gan_loss = tf.math.reduce_mean(1.0 - disc_sr)
    gan_loss = tf.math.reduce_mean((tf.ones_like(disc_sr) - tf.random.uniform(disc_sr.shape) * 0.1) - disc_sr)
    gan_loss = tf.clip_by_value(gan_loss, 0.0, 1.0)
    
    # We also want the image to look as similar as possible to the HR images
    sim_loss = similarity_loss(hr, sr)
    
    # Lambda weights how much we value each loss
    return gan_loss + (lambda_ * sim_loss)


@tf.function
def predict_and_compute_loss_gan(hr, outputs, generator, discriminator):
    # Get generator output
    sr = generator(lr, training=True)

    # Get discriminator output for the real image and the super resolved image
    disc_real_output = discriminator(hr, training=True)
    disc_generated_output = discriminator(sr, training=True)

    # Compute losses for each network using above functions
    gen_loss = generator_loss(disc_generated_output, sr, hr)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    return gen_loss, disc_loss

        
@tf.function
def train_step_gan(lrs, hr, generator, discriminator, generator_optimizer, discriminator_optimizer, gen_train = False):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        sr = generator(lrs)
        
        hr_, sr_ = apply_mask(hr, sr)
        
        # Todo add smoothing
        disc_hr = discriminator(hr_, training=True)
        disc_sr = discriminator(sr_, training=True)
        
        disc_loss = discriminator_loss(disc_hr, disc_sr)
        gen_loss = generator_loss(disc_sr, hr_, sr_)
        
        
        # We compute gradient for each part
        
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        if tf.equal(gen_train, True):
            generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        
    return gen_loss, disc_loss


'''
==============================
        Visualization
==============================
'''
        
def show_pred(model, lrs, hr, max_lr=35):
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
    
    
def save_pred(model, lrs, hr, dir_name, epoch, max_lr=35):
    predicted = model.predict(lrs)
    fig = plt.figure(figsize=(15,15))
    
    display_list = [lrs[0][:,:,0], predicted[0][:,:,0], hr[0][:,:,0]]
    title = ['Input Image (1 of ' + str(max_lr) + ')', 'Predicted Image: Epoch' + str(epoch), 'Real image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
        
    # Make sure dir is created
    base_path = "Result/" + dir_name + "/"
    os.makedirs(base_path, exist_ok=True)
    plt.savefig(base_path + str(epoch) + ".png", bbox_inches='tight')
    plt.close(fig)
    