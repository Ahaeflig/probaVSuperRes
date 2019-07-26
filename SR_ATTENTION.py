from tensorflow.keras.layers import Conv2D, Conv3D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input, Add, UpSampling2D
import tensorflow as tf

from ops import residual_block_gen, up_sample_block, residual_attention_block

class SRAttention():
    """ Defines a DL model for Super Resolution built from an attention mechanism and residual blocks
    
    Args:
        channel_dim: Number of LRs used
        number_residual_block: Number of residual block used in the network
    
    """
    
    
    def __init__(self, channel_dim = 9, number_attention_residual_block = 3, include_batch_norm = False):
        inputs = Input(shape=[128, 128, channel_dim])
        
        self.number_residual_block = number_residual_block
        self.batch_norm = include_batch_norm
        self.model = self.build_model(inputs)
        self.name = "ResidualCNN"
        
    
    def build_model(self, inputs):
        """ Builds the model with all the pieces put together:
        1) General Conv + activation
        2) n Residual Block (default)
        3) General Conv + batch norm + skip connection
        4) Upsample Block
        5) Final Convulation + activation to generate HR
        
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

        skip_connection = model
        
        # residual attentions
        for i in range(self.number_residual_block):
            model = residual_attention_block(model, 3, 128, 1, self.batch_norm)

        model = Conv3D(128, kernel_size=3, strides = 1, padding = "SAME")
        model = LeakyReLU(alpha = 0.2)(model)
            
        # upscale
        model = up_sample_block(model, 3, 128, 1, size=3, skip=None)
        
        model = Conv3D(128, kernel_size=3, strides = 1, padding = "SAME")
        model = Conv3D(64, kernel_size=3, strides = 1, padding = "SAME")
        LeakyReLU(alpha = 0.2)(model)
        
        model = Conv2D(1, kernel_size = 3, strides = 1, padding = "same", kernel_initializer='he_normal')(model)
        
        return tf.keras.Model(inputs = inputs, outputs = model)     