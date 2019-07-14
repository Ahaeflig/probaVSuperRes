import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

import os

"""
Helper file with functions written to perform the optimization of a neural net.
"""



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
def compute_loss(lrs, hr, model):
    output = model(lrs, training=True)
    hr_masked, output_masked = applyMask(hr, output) 
    return clearMSE(hr_masked, output_masked) + clearMAE(hr_masked, output_masked)


@tf.function
def train_step(lrs, hr, model, optimizer):
    with tf.GradientTape() as model_tape:
        loss = compute_loss(lrs, hr, model)
        # Compute gradient for parameter
        model_gardients = model_tape.gradient(loss, model.trainable_variables)
        # Apply the gradient to the variables, model weights are updated here
        optimizer.apply_gradients(zip(model_gardients, model.trainable_variables))
        
    return loss
        
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
    