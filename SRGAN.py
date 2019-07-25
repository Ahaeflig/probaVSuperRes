from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input
import tensorflow as tf
from tensorflow import keras

from ops import discriminator_block


class SRGAN():
    
    def __init__(self, generator_model, channel_dim = 5, include_batch_norm = False):
        gen_input = tf.keras.layers.Input(shape=[128, 128, channel_dim])
        self.batch_norm = include_batch_norm
        
        self.generator = generator_model
        
        gen_out = self.generator(gen_input)
        
        self.discriminator = self.build_discriminator()
        
        model_out = self.discriminator(gen_out)

        self.model = tf.keras.Model(gen_input, [gen_out, model_out])
      
    
    def build_discriminator(self):
        discriminator_input = tf.keras.layers.Input(shape=[384,384,1])
        
        model = Conv2D(32, 3, strides=1, padding='same')(discriminator_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 32, 3, 2, self.batch_norm)
        model = discriminator_block(model, 64, 3, 1, self.batch_norm)
        model = discriminator_block(model, 64, 3, 2, self.batch_norm)
        model = discriminator_block(model, 128, 3, 2, self.batch_norm)
        model = discriminator_block(model, 256, 3, 2, self.batch_norm)
        
        model = tf.keras.layers.GlobalAveragePooling2D()(model)
        model = Flatten()(model)
        
        model = Dense(128)(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        # One final output encoding if the image was "generated" aka upsampled or not
        model = Dense(1)(model)
        model = Activation('sigmoid')(model)
        
        return tf.keras.Model(inputs = discriminator_input, outputs = model)
    
    

