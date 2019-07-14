from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input, Add, UpSampling2D
import tensorflow as tf

class MultipleResidualCNN():
    
    def __init__(self, channel_dim = 35):
        inputs = Input(shape=[128, 128, channel_dim])
        self.model = self.build_model(inputs)
        self.name = "ResidualCNN"
    
    @staticmethod
    def residual_block_gen(model, kernel_size, filters, strides):
        previous = model

        model = Conv2D(filters, kernel_size, strides=strides, padding='same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(filters, kernel_size, strides=strides, padding='same')(model)
        model = BatchNormalization()(model)

        return Add()([previous, model])
    
    @staticmethod
    def up_sample_block(model, kernel_size, filters, strides):
        model = Conv2D(filters, kernel_size, strides, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D(size = 3)(model)
        model = Conv2D(filters, kernel_size, strides, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)

        return model
    
    def build_model(self, inputs):
        model = Conv2D(64, 9, strides=1, padding='same')(inputs)
        model = LeakyReLU(alpha = 0.2)(model)
        
        skip_connection = model
        
        # Residual Blocks, 3 should probably be increased on faster machines
        for i in range(3):
            model = self.residual_block_gen(model, 3, 64, 1)

        model = Conv2D(filters = 1, kernel_size = 9, strides = 1, padding = "same")(model)
        model = BatchNormalization()(model)
        model = Add()([model, skip_connection])
        
        # Upsample
        model = self.up_sample_block(model, 3, 256, 1)
        model = Conv2D(1, kernel_size = 9, strides = 1, padding = "same")(model)
        model = Activation('sigmoid')(model)

        return tf.keras.Model(inputs = inputs, outputs = model)     