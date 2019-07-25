""" Collection of deep learning blocks and operations to build models
"""

from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input, Add, UpSampling2D
import tensorflow as tf

    
def discriminator_block(model, filters, kernel, strides, batch_norm):
    model = Conv2D(filters, kernel, strides = strides, padding = "same", kernel_initializer='he_normal')(model)
    if batch_norm:
        model = BatchNormalization()(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model


def residual_block_gen(model, kernel_size, filters, strides, batch_norm):
    """ ResNet residual block a bit modified with a LeakyReLU and no activation function after the additon

        Args:
            model: the current model
            kernel_size: size of the kernel used for convolutions
            filters: last dimension of the output
            strides: shift amount between each convolution
    """

    previous = model

    model = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(model)

    if batch_norm:
        model = BatchNormalization()(model)

    model = LeakyReLU(alpha = 0.2)(model)
    model = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(model)

    if batch_norm:
        model = BatchNormalization()(model)

    return Add()([previous, model])


def up_sample_block(model, kernel_size, filters, strides, size=2, skip=None):
    """ Up Sample block based on Convolution, LeakyRelu and UpSampling2D (repeats the rows and columns)

        Args:
            model: the current model
            kernel_size: size of the kernel used for convolutions
            filters: last dimension of the output
            strides: shift amount between each convolution
    """

    if skip is not None:
        model = Concatenate(axis=-1)([model, skip])

    model = Conv2D(filters, kernel_size, strides, padding = "same", kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha = 0.2)(model)
    model = UpSampling2D(size = size)(model)
    model = Conv2D(filters, kernel_size, strides, padding = "same", kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model
    
    
def down_sample_block(model, kernel_size, filters, batch_norm):

    skip = model

    model = self.residual_block_gen(model, 3, filters, 1, batch_norm)
    model = Conv2D(filters, kernel_size, (2,2), padding = "valid", kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model, skip
    
    
    
