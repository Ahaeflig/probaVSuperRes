from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input
import tensorflow as tf
from tensorflow import keras


class SuperResolutionGan():
    
    def __init__(self, generator_path):
        
        gen_input = tf.keras.layers.Input(shape=[128,128,35])
        
        self.discriminator = self.build_discriminator()
        self.generator = keras.load_model(generator_path)
        
        gen_out = self.generator(gen_input)
        
        # Discriminator use pre-trained model for feature extraction that expects 3 channel images
        gen_out_3channel = tf.tile(gen_out, [1, 1, 1, 3])
        model_out = self.discriminator(gen_out_3channel)

        self.model = tf.keras.Model(inputs, [gen_out, model_out])
      
    
    def build_discriminator(self):
        
        discriminator_input = tf.keras.layers.Input(shape=[384,384,3])
        
        # Use Pre-trained network to extract features
        base_model = tf.keras.applications.MobileNetV2(input_shape=(384,384, 1), include_top=False, weights='imagenet')
        base_model.trainable = False
        
        model = tf.keras.layers.GlobalAveragePooling2D()(base_model)
        
        model = Dense(256)(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        # One final output encoding if the image was "generated" aka upsampled or not
        model = Dense(1)(model)
        model = Activation('sigmoid')(model)
        
        return tf.keras.Model(inputs = discriminator_input, outputs = model)
        
    '''
    def build_discriminator(self):
        discriminator_input = tf.keras.layers.Input(shape=[384,384,1])
        
        model = Conv2D(32, 3, strides=1, padding='same')(discriminator_input)
        model = LeakyReLU(alpha = 0.2)(model)
        model = self.discriminator_block(model, 32, 3, 2)
        model = self.discriminator_block(model, 64, 3, 1)
        model = self.discriminator_block(model, 64, 3, 2)
        model = self.discriminator_block(model, 128, 3, 1)
        model = self.discriminator_block(model, 128, 3, 2)
        model = self.discriminator_block(model, 256, 3, 1)
        model = self.discriminator_block(model, 256, 3, 2)
        # need more depth?
        
        model = Flatten()(model)
        model = Dense(512)(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        # One final output encoding if the image was "generated" aka upsampled or not
        model = Dense(1)(model)
        model = Activation('sigmoid')(model)
        
        return tf.keras.Model(inputs = discriminator_input, outputs = model)
    
    
    def discriminator_block(self, model, filters, kernel, strides):
        model = Conv2D(filters, kernel, strides = strides, padding = "same", kernel_initializer=initializer)(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)

        return model
    '''