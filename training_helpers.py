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

def applyMask(hr, generated):
    """ Finds np.NaN values in the HR image and set those pixel to 0 (False) in hr and generated.
    
        Args:
            hr: High resolution image where obstructed pixel are encoded as nan values
            generated: generated image by the model which
            
        Return:
            The modified (masked) HR and generated images
    
    """

    obs = tf.where(tf.math.is_nan(hr), False, True)
    hr_ = tf.boolean_mask(hr, obs)
    generated_ = tf.boolean_mask(generated, obs)
    return hr_, generated_


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


def clearMAE(hr_masked, generated_masked):
    """ MAE loss in the same vein as clearMSE
    
    Args:
        hr_masked: the masked HR image
        generated_masked: the masked generated image
        
    Return:
        The mean average error between both image with a corrected bias
    """
    
    bias = tf.math.reduce_mean(hr_masked - generated_masked) 
    loss = tf.math.reduce_mean(hr_masked - (generated_masked + bias))
    return loss


@tf.function
def compute_loss(lrs, hr, model, training=True):
    sr = model(lrs, training)
    hr_masked, sr_masked = applyMask(hr, sr) 
    return clearMSE(hr_masked, sr_masked) + clearMAE(hr_masked, sr_masked)


@tf.function
def train_step(lrs, hr, model, optimizer):
    with tf.GradientTape() as model_tape:
        loss = compute_loss(lrs, hr, model)
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

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = disc_real_output
    generated_loss =  tf.math.abs(disc_generated_output - 1.0)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_output, sr, hr, lambda_ = 10):
    # we try to trick the discriminator to predict our generated image to be considered as valid ([1])
    gan_loss = 1.0 - disc_output
    
    # We also want the image to look as similar as possible to the HR images
    hr_masked, output_masked = applyMask(hr, sr) 
    clear_losses = clearMSE(hr_masked, output_masked) + clearMAE(hr_masked, output_masked)
    
    # Lambda weights how much we value each loss
    return gan_loss + (lambda_ * clear_losses)


@tf.function
def compute_loss_gan(lr, hr, generator, discriminator):
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
def train_step_gan(lr, hr, generator, discriminator, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_loss, disc_loss = compute_loss(lr, hr, generator, discriminator)
        # We compute gradient for each part
        generator_gradients = gen_tape.gradient(gen_loss, srg.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, srg.discriminator.trainable_variables)

        # We apply the gradient to the variables, tf2.0 way
        generator_optimizer.apply_gradients(zip(generator_gradients, srg.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, srg.discriminator.trainable_variables))
        
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
    