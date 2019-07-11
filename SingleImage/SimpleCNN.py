from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input
import tensorflow as tf

# Take all patches and upscale
class SimpleCNN():
    
    def __init__(self):
        inputs = Input(shape=[384,384,1])
        self.simpleCNN = self.build_model(inputs)
        
    def residual_block_gen(self, model, kernel_size, filters, strides):
        previous = model

        initializer = tf.random_normal_initializer(0., 0.02) # TODO

        model = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer)(model)
        model = BatchNormalization()(model)
        model = LeakyReLU()(model)
        model = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer)(model)
        model = BatchNormalization()(model)

        return tf.keras.layers.Add()([previous, model])
    
    def build_model(self, inputs):
        model = Conv2D(64, 9, strides=1, padding='same')(inputs)

        for i in range(3):
            model = self.residual_block_gen(model, 3, 64, 1)

        model = Conv2D(filters = 1, kernel_size = 9, strides = 1, padding = "same")(model)
        model = Activation('sigmoid')(model)

        return tf.keras.Model(inputs = inputs, outputs = model)
            
                
            
            
            
            