from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input, Add, UpSampling2D
import tensorflow as tf

class SRCNN():
    """ Defines a DL model for Super Resolution that uses a residual blocks and an up sampling block to super resolve multiple low resolution images stacked together
    
    Args:
        channel_dim: Number of LRs used
        number_residual_block: Number of residual block used in the network
    
    """
    
    
    def __init__(self, channel_dim = 35, number_residual_block = 3, include_batch_norm = False):
        inputs = Input(shape=[128, 128, channel_dim])
        
        self.number_residual_block = number_residual_block
        self.batch_norm = include_batch_norm
        self.model = self.build_model(inputs)
        self.name = "ResidualCNN"
        
    
    @staticmethod
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
    
    
    @staticmethod
    def up_sample_block(model, kernel_size, filters, strides):
        """ Up Sample block based on Convolution, LeakyRelu and UpSampling2D (repeats the rows and columns)
        
            Args:
                model: the current model
                kernel_size: size of the kernel used for convolutions
                filters: last dimension of the output
                strides: shift amount between each convolution
        """
        
        model = Conv2D(filters, kernel_size, strides, padding = "same", kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = UpSampling2D(size = 3)(model)
        model = Conv2D(filters, kernel_size, strides, padding = "same", kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)

        return model
    
    
    def build_model(self, inputs):
        """ Builds the model with all the pieces put together:
        1) General Conv + activation
        2) 3 Residual Block (default)
        3) General Conv + batch norm + skip connection
        4) Upsample Block
        5) Final Convulation + activation to generate HR, aka filters = 1
        
        Args:
            inputs: Keras.layers.input input shape
        
        Returns:
            a tf.keras.Model, the built model
        """
        model = Conv2D(32, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(32, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(128, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(128, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)

        skip_connection = model
        
        for i in range(self.number_residual_block):
            model = self.residual_block_gen(model, 3, 128, 1, self.batch_norm)

        model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        if self.batch_norm:
            model = BatchNormalization()(model)
            
        model = Add()([model, skip_connection])
        
        # Upsample
        model = self.up_sample_block(model, 3, 128, 1)
        
        model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same")(model)
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(model)
        model = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        
        model = Conv2D(1, kernel_size = 9, strides = 1, padding = "same", kernel_initializer='he_normal')(model)
        
        model = Activation('sigmoid')(model)

        return tf.keras.Model(inputs = inputs, outputs = model)     