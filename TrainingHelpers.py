import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

def applyMask(hr, generated):
    obs = tf.where(tf.math.is_nan(hr), False, True)
    clr = obs
    hr_ = tf.boolean_mask(hr, clr)
    generated_ = tf.boolean_mask(generated, clr)
    return hr_, generated_

def cMSE(hr_masked, generated_masked):
    # calculate the bias in brightness b
    pixel_diff = hr_masked - generated_masked
    b = K.mean(pixel_diff)

    # calculate the corrected clear mean-square error
    pixel_diff -= b
    cMse = K.mean(pixel_diff * pixel_diff)

    return cMse


def cMAE(hr_masked, generated_masked):
    pixel_diff = hr_masked - generated_masked
    b = K.mean(pixel_diff)
    
    pixel_diff -= b
    return K.mean(pixel_diff)

def clearMSE(hr_masked, generated_masked):
    """As defined in https://kelvins.esa.int/proba-v-super-resolution/scoring/
    
        MSE loss that takes into account brightness and stuff TODO
    """
    bias = tf.math.reduce_mean(generated_masked - hr_masked) 
    loss = tf.math.reduce_mean(tf.pow(hr_masked - (generated_masked - bias), 2))
    return loss


def clearMAE(hr_masked, generated_masked):
    """As defined in https://kelvins.esa.int/proba-v-super-resolution/scoring/
    
        MSE loss that takes into account brightness and stuff TODO
    """
    bias = tf.math.reduce_mean(generated_masked - hr_masked) 
    loss = tf.math.reduce_mean(hr_masked - (generated_masked - bias))
    return loss


@tf.function
def compute_loss(lr, hr, model):
    output = model(lr, training=True)
    hr_masked, output_masked = applyMask(hr, output) 
    return clearMSE(hr_masked, output_masked) + clearMAE(hr_masked, output_masked)


@tf.function
def train_step(lr, hr, model, optimizer):
    with tf.GradientTape() as model_tape:
        loss = compute_loss(lr, hr, model)
        # We compute gradient for each part
        model_gardients = model_tape.gradient(loss, model.trainable_variables)
        # We apply the gradient to the variables, tf2.0 way
        optimizer.apply_gradients(zip(model_gardients, model.trainable_variables))
        
        
def show_pred(model, lrs, hr):
    predicted = model.predict(lrs)
    plt.figure(figsize=(15,15))
    
    display_list = [lrs[0][:,:,0], predicted[0][:,:,0], hr[0][:,:,0]]
    title = ['Input Image (1 of 9)', 'Predicted Image', 'Real image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
    