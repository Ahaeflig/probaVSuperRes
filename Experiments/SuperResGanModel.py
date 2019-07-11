from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Flatten, Dense, Input
import tensorflow as tf

class SuperResolutionGan():
    
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        inputs = Input(shape=[384,384,1])
        gen_out = self.generator(inputs)
        model_out = self.discriminator(gen_out)

        self.srg = tf.keras.Model(inputs, [gen_out, model_out])
        
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
        
        # One final output value if the image was "generated" aka upsampled or not
        model = Dense(1)(model)
        model = Activation('sigmoid')(model)
        
        return tf.keras.Model(inputs = discriminator_input, outputs = model)
        
    def build_generator(self):
            generator_input = tf.keras.layers.Input(shape=[384,384,1])

            model = Conv2D(64, 9, strides=1, padding='same')(generator_input)
            for i in range(10):
                model = self.residual_block_gen(model, 3, 64, 1)

            model = Conv2D(32, 3, strides=1, padding='same')(model)
            model = LeakyReLU(alpha = 0.25)(model)
            #model = BatchNormalization()(model) # momentum = 0.5?
            model = Conv2D(16, 3, strides=1, padding='same')(model)
            model = LeakyReLU(alpha = 0.25)(model)
            model = Conv2D(1, 9, strides=1, padding='same')(model)
            model = Activation('tanh')(model)

            return tf.keras.Model(inputs=generator_input, outputs=model)

    def residual_block_gen(self, model, kernel_size, filters, strides):
        # for skip connection since this is a residual block
        previous = model
        
        initializer = tf.random_normal_initializer(0., 0.02) # TODO
        
        model = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer)(model)
        model = BatchNormalization()(model)
        model = LeakyReLU()(model)
        model = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=initializer)(model)
        model = BatchNormalization()(model)
        
        return tf.keras.layers.Add()([previous, model])
    
    
    def discriminator_block(self, model, filters, kernel, strides):
        initializer = tf.random_normal_initializer(0., 0.02)
        
        model = Conv2D(filters, kernel, strides = strides, padding = "same", kernel_initializer=initializer)(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)

        return model