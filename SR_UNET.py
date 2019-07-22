from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input, Add, UpSampling2D
import tensorflow as tf

class SR_UNET():
    """ Defines a DL model for Super Resolution that uses a residual blocks and an up sampling block to super resolve multiple low resolution images stacked together
    
    Args:
        channel_dim: Number of LRs used
        number_residual_block: Number of residual block used in the network
    
    """
    
    
    def __init__(self, channel_dim = 5, include_batch_norm = False):
        inputs = Input(shape=[128, 128, channel_dim])
        self.batch_norm = include_batch_norm
        self.model = self.build_model(inputs)
        self.name = "UNET"
        
    
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
    
    
    @staticmethod
    def down_sample_block(model, kernel_size, filters, batch_norm):
        
        skip = model
        
        model = self.residual_block_gen(model, 3, filters, 1, batch_norm)
        model = Conv2D(filters, kernel_size, (2,2), padding = "valid", kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        return model, skip
    
    
    def build_model(self, inputs):
        """ Builds the model with all the pieces put together
     
        Args:
            inputs: Keras.layers.input input shape
        
        Returns:
            a tf.keras.Model, the built model
        """
        
        model = UpSampling2D(size = 3)(inputs)
        model = Conv2D(32, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        if self.batch_norm:
            model = BatchNormalization()(model)
        model = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        model = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        skip_head = model
        
        # Downsample and store 3 times 384 -> 192 -> 96 -> 192 -> 384
        
        model, skip_0 = self.down_sample_block(model, 3, 64, self.batch_norm)
        model, skip_1 = self.down_sample_block(model, 3, 128, self.batch_norm)
        model, skip_2 = self.down_sample_block(model, 3, 128, self.batch_norm)
        model, skip_3 = self.down_sample_block(model, 3, 256, self.batch_norm)
        
        model = self.residual_block_gen(model, 3, 512, 1, batch_norm)
        
        model = self.up_sample_block(model, 3, 256, 1, size=2, skip=skip_3)
        model = self.up_sample_block(model, 3, 128, 1, size=2, skip=skip_2)
        model = self.up_sample_block(model, 3, 128, 1, size=2, skip=skip_1)
        model = self.up_sample_block(model, 3, 64, 1, size=2, skip=skip_0)
        
        if self.batch_norm:
            model = BatchNormalization()(model)
            
        model = Add()([model, skip_head])
        
        model = Conv2D(64, kernel_size = 3, strides = 1, padding = "same", kernel_initializer='he_normal')(model)
        model = Conv2D(64, kernel_size = 3, strides = 1, padding = "same", kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Conv2D(32, kernel_size = 3, strides = 1, padding = "same", kernel_initializer='he_normal')(model)
        model = Conv2D(32, kernel_size = 3, strides = 1, padding = "same", kernel_initializer='he_normal')(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        if self.batch_norm:
            model = BatchNormalization()(model)
            
        x = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        model = Activation('sigmoid')(model)

        return tf.keras.Model(inputs = inputs, outputs = model)     