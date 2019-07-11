import os
from glob import glob

import skimage
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from supreshelper import *

from typing import List

PATCH_SIZE = 32
LR_HR_RATIO = 3
HR_MAX = 17500


class Scene() :
    """Data holder for scene images paths containing list of paths and paths to all
    relevant images in the scene
    """
    def __init__(self, qms: List[str], lrs: List[str], sm: str, uplr:str, hr: str=None):
        self.qms = qms
        self.lrs = lrs
        self. sm = sm
        self.hr = hr
        self.uplr = uplr

class ProbaVDataset():
    """Management of the proba-v-super-resolution challenge's data.
    1) Check if the provided data matches the hypothesis
    2) Load images/patches
    3) ...
    
    Attributes:
        data_dir: string path to the Data folder (containing the train and test folders)
    """
    
    def __init__(self, data_dir: str):
        self.train_samples = glob(data_dir +  "train/*/*") 
        self.test_samples = glob(data_dir + "test/*/*")
        
        self._check_dataset()
        
        # We store current pivot to vizualize, a bit hacky for now
        self.pivot = None
        
    
    def _check_dataset(self):
        """Check that the downloaded data respects the following
            - always with at least 9 QM/LR
            - Same number of LR and QM
            - SM/HR exists
        """
        for scene_path in self.train_samples + self.test_samples:
            scene = self.get_scene_content(scene_path)
            assert (len(scene.qms) >= 9), "Less than 9 images at scene" + scene_path
            assert (os.path.isfile(scene.sm)), "Missing SM image at scene " + scene_path
            assert (len(scene.lrs) == len(scene.qms) ), "Number of QMS and LRS  images are not identical at scene " + scene_path + ": " + str(len(scene.lrs)) + " / " + str(len(scene.qms))
            
            if scene_path.split("/")[1] == "train":
                assert (os.path.isfile(scene.hr)), "Missing HR image at scene " + scene_path
            else:
                assert (os.path.isfile(scene.uplr)), "Missing up_LR image at scene " + scene_path
                
    def get_scene_content(self, scene_path: str):
        """Finds the file for a given scene (folder) in the dataset

            Args:
                scene_path: The path to scene's folder

            Returns
                a Scene object (see Scene class)
        """
        qms = glob(scene_path + "/QM*.png")
        lrs = glob(scene_path + "/LR*.png")
        sm = scene_path + "/SM.png"
        hr = scene_path + "/HR.png"
        up_lr = scene_path + "/up_LR.tfrecords"
        
        return Scene(qms, lrs, sm, up_lr, hr)
    
    
    """
    Patches Block
    """
    def load_image(self, img_path, is_quality_map=False):
        image = tf.image.decode_png(tf.io.read_file(img_path), dtype=tf.uint16)
        input_image = tf.cast(image, tf.float32)
        if not is_quality_map:
            return input_image / HR_MAX
        else:
            return input_image
        
    # Jitter is basically 40 - 32
    def load_qms_lrs_patches(self, scene: Scene, pivot: tf.Tensor, jitter: int=8):
        j_d = (int)((PATCH_SIZE + 8) / 2)
        h,w = pivot
        
        slices = [slice(h - j_d, h + j_d), slice(w - j_d, w + j_d), 0]
        qm_tensors = [self.load_image(path, True)[slices] for path in scene.qms]
        lr_tensors = [self.load_image(path)[slices] for path in scene.lrs]

        # Contains list of [qm, lr] tensors pairs 
        patches_stack = [tf.stack([qm, lr], axis=0) for qm, lr in zip(qm_tensors, lr_tensors)]
        return patches_stack
    
    def load_hr_sm(self, scene: Scene):
        hr = self.load_image(scene.hr)
        sm = self.load_image(scene.sm, True)
        return sm, hr
    
    def visuzalize(self, scene: Scene):
        """ Shows an example of input output 
        """
        # Get scene tensors
        stack_cropped, label = self.get_random_patch_stack_and_label(scene)
        
        fig = plt.figure(figsize=(25,25))
        i = 1
        for s_c in stack_cropped:
            ax = plt.subplot(15,4,i)
            ax.axis('off');
            plt.imshow(s_c[0], cmap="copper", vmin=0, vmax=255)
            ax = plt.subplot(15,4,i+1)
            ax.axis('off');
            plt.imshow(s_c[1])
            i += 2
            
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(1,2,1)
        ax.axis('off');
        plt.imshow(label[0], cmap="copper", vmin=0, vmax=255)
        ax = plt.subplot(1,2,2)
        ax.axis('off');
        plt.imshow(label[1])
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,2,1)
        ax.axis('off');
        ax.add_patch(matplotlib.patches.Rectangle((self.pivot[1]*3 - 48, self.pivot[0]*3 - 48), 96, 96, color="red", alpha=0.3))
        plt.imshow(self.load_image(scene.sm, True)[:,:,0], cmap="copper", vmin=0, vmax=255)
        
        ax = fig.add_subplot(1,2,2)
        ax.axis('off');
        plt.imshow(self.load_image(scene.hr)[:,:,0])
        ax.add_patch(matplotlib.patches.Rectangle((self.pivot[1]*3 - 48, self.pivot[0]*3 - 48), 96, 96, color="red", alpha=0.3))
        plt.show()
    
    def get_random_patch_stack_and_label(self, scene: Scene):
        pivot = ds.get_pivot().numpy()
        self.pivot = pivot
        
        jitter = 8
        patches_stack = self.load_qms_lrs_patches(scene, pivot, jitter)
        # Get Random crop from 40 (depends on jitter) by 40 patch
        stack_cropped = [tf.image.random_crop(patch, size=[2, PATCH_SIZE, PATCH_SIZE]) for patch in patches_stack]
        
        # Get label at pivot point
        sm, hr = self.load_hr_sm(scene)
        j_d = (int)(PATCH_SIZE * LR_HR_RATIO / 2)
        h,w = pivot * 3
        
        slices = [slice(h - j_d, h + j_d), slice(w - j_d, w + j_d), 0]
        label = tf.stack([sm[slices], hr[slices]], axis=0)
        return stack_cropped, label
    
    def get_pivot(self, grid_size:int =128, border_size: int=40):
        return tf.random.uniform([2], border_size, grid_size-border_size, dtype=tf.dtypes.int32)

    """
    Singe Image Block
    """
    def load_data(self, path: str):
        """ Load data stored in .npy format
            
            Args:
                path: Path to the .npy file
                
            Return:
                loaded numpy array as a Tensor
        """
        data = np.load(path)
        return tf.expand_dims(tf.convert_to_tensor(data, dtype=tf.float32), axis=2)
    
    @tf.function
    def load_train_image(self, lr_hr_sm, augment=True):
        lr_hr_sm = tf.expand_dims(lr_hr_sm, axis=2)
        
        w = tf.shape(lr_hr_sm)[1]
        w = w // 3
        lr = lr_hr_sm[:, :w, :]
        hr = lr_hr_sm[:, w:2*w, :]
        sm = lr_hr_sm[:, 2*w:3*w, :]
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        sm = tf.cast(sm, tf.float32)
        
        if augment:
            return self.random_jitter(lr, hr, sm)
        else:
            return lr, self.apply_mask(hr, sm)
    
    def apply_mask(self, hr, sm):
        return  tf.where(sm > 0, hr, np.NaN)
    
    def resize(self, image, height, width):
        return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def random_crop(self, lr, hr, sm):
        stacked_image = tf.stack([lr, hr, sm], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[3, 384, 384, 1])
        return cropped_image[0], cropped_image[1], cropped_image[2]
        
    @tf.function()
    def random_jitter(self, lr, hr, sm):
        lr, hr, sm = self.resize(lr, 429, 429), self.resize(hr, 429, 429), self.resize(sm, 429, 429)
        lr, hr, sm = self.random_crop(lr, hr, sm)

        if tf.random.uniform(()) >= 0.5:
            lr = tf.image.flip_left_right(lr)
            hr = tf.image.flip_left_right(hr)
            sm = tf.image.flip_left_right(sm)
            
        if tf.random.uniform(()) >= 0.5:
            lr = tf.image.flip_up_down(lr)
            hr = tf.image.flip_up_down(hr)
            sm = tf.image.flip_up_down(sm)
        
        return lr, self.apply_mask(hr, sm)
    
    def load_test_image(self, filepath):
        return self.load_data(filepath)
        