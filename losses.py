import tensorflow as tf

class Losses():
    """ Collection of losses used throughout the project and utils to compute the loss
    """
    
    @staticmethod
    def cMSE(hr_masked, generated_masked):
        """As defined in https://kelvins.esa.int/proba-v-super-resolution/scoring/
        MSE loss that is not sensitive to average difference

        Args:
            hr_masked: the masked HR image
            generated_masked: the masked generated image

        Return:
            The mean squared error between both image with a corrected bias

        """

        bias = tf.math.reduce_mean(hr_masked - generated_masked) 
        loss = tf.math.reduce_mean(tf.pow(hr_masked - (generated_masked + bias), 2))
        return loss
    
    
    @staticmethod
    def MAE(hr_masked, sr_masked):
        return tf.math.reduce_mean(tf.losses.mean_absolute_error(hr_masked, sr_masked))    
    
    
    @staticmethod
    def MSE(hr_masked, sr_masked):
        return tf.math.reduce_mean(tf.losses.mean_squared_error(hr_masked, sr_masked)) 
    
    
    @staticmethod
    def cMAE(hr_masked, generated_masked):
        """ MAE loss in the same vein as clearMSE

        Args:
            hr_masked: the masked HR image
            generated_masked: the masked generated image

        Return:
            The mean average error between both image with a corrected bias
        """

        bias = tf.math.reduce_mean(hr_masked - generated_masked) 
        loss = tf.math.reduce_mean(tf.abs(hr_masked - (generated_masked + bias)))
        return loss
    
    
    @staticmethod
    def log10(x):
        """ Compute base 10 log
        """
        numerator = tf.math.log(x)
        # tf.math.log(tf.constant(10.0))
        denominator = 2.3025851
        return numerator / denominator

    
    @staticmethod
    def cPSNR(hr, sr):
        """ clear Peak Signal to Noise Ratio loss as defined here:
            https://kelvins.esa.int/proba-v-super-resolution/scoring/
        """
        # 1e-10: makes it such that if the cMSE is 0, then we get the best score defined as 100
        return -10.0 * Losses.log10(Losses.cMSE(hr, sr) + 1e-10)
    
    
    @staticmethod
    def cPSNR_metric(hr, sr):
        """ Apply the mask itself and calls cPSNR
        """
        
        hr_, sr_= Losses.apply_mask(hr, sr)
        return Losses.cPSNR(hr_, sr_)
        
        
    @staticmethod
    def SSIM(hr, sr):
        # TODO what would be nice for sigma?
        diff_loss = 1.0 - tf.image.ssim(hr, sr, 1.0, filter_size=3, filter_sigma=0.9, k1=0.01, k2=0.03)
        return tf.math.reduce_mean(diff_loss)
    
    
    @staticmethod
    def MS_SSIM(hr, sr):
        # Multiscale SSIM
        diff_loss = 1.0 - tf.image.ssim_multiscale(hr, sr, 1.0, filter_size=3, filter_sigma=1.2, k1=0.01, k2=0.03)
        return tf.math.reduce_mean(diff_loss)
    
    
    @staticmethod
    def apply_mask(hr, sr):
        """ Finds np.NaN values in the HR image and set those pixel to 0 (False) in hr and generated.

            Args:
                hr: High resolution image where obstructed pixel are encoded as nan values
                sr: generated image by the model which

            Return:
                The modified (masked) HR and generated images

        """
        hr_ = tf.where(tf.math.is_nan(hr), 0.0, hr)
        sr_ = tf.where(tf.math.is_nan(hr), 0.0, sr)

        return hr_, sr_