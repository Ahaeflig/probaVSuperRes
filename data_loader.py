from glob import glob
import tensorflow as tf

class MultipleDataLoader():
    """ Class that provdes utilities to read and process TFRecords that contains multiple LR images and their correpsonding HR image.
        The main point of the class is to map TFRecords to the parse function to build a dataset.
        This will turn the TF dataset into an iterator yielding Tensors ready to be fed to the model.
        
        Define the loader:
        loader = MultipleDataLoader("DataTFRecords/train/", 4, True, True, 5)
        
        Iterate over the data:
        for lrs, hr in loader().take(1):
            ...
        
        Args:
            data_dir: Path to the folder containg TFRecords, typically: "DataTFRecords/train/"
            batch_size: Batch size of the created dataset
            repeat: Should the dataset be repeated (Used when training with model.fit())
            augment: Apply data augmentations
            num_lrs: Number of LRs in the TFRecords (generated with generate_tfrecords.py)
        
    """
    
    
    def __init__(self, data_dir: str, batch_size: int, repeat: bool, augment:bool, shuffle: bool, num_lrs: int):
        self.files = glob(data_dir +  "*/*/multiple.tfrecords")
        
        assert(len(self.files) > 1), "No scenes found, check path to train folder"
        
        self.batch_size = batch_size
        self.repeat = repeat
        self. augment = augment
        self.shuffle = shuffle
        self.num_lrs = num_lrs
        
        self.dataset = self.build_dataset()
    
    def __call__(self):
        return self.dataset
    
    
    def build_dataset(self):
        """ Builds the TFRecordDataset that can be iterated over when training models.
        
        """
        
        # Create a tf dataset
        tf_dataset = tf.data.TFRecordDataset(self.files)
        
        # For batch_size = 1, having multiple workers can make the tf2 crash
        num_parallel = tf.data.experimental.AUTOTUNE if self.batch_size > 1 else 1
            
        tf_dataset = tf_dataset.map(lambda x: self.parse_multiple_fixed(x), num_parallel_calls=num_parallel)

        if self.repeat:
            tf_dataset = tf_dataset.repeat() 
        
        if self.shuffle:
            tf_dataset = tf_dataset.shuffle(len(self.files))
            
        tf_dataset = tf_dataset.batch(self.batch_size)
        
        return tf_dataset
    
    
    def get_shuffled_copy(self):
        """ Returns a shuffled copy of the dataset, useful when manually iterating over the dataset,
            we need to manually shuffle as well.
        
        """
        dataset_epoch = self.dataset.shuffle(len(self.files))
        return dataset_epoch
        
        
    @tf.function
    def parse_multiple_fixed(self, example_proto):
        """ Parses a tf record and reconstruct the correct data from the following features:
        
            feature ={"lrs": tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate(lrs).ravel())),
                      "hr": tf.train.Feature(float_list=tf.train.FloatList(value=hr.flatten()))}
                      
            where lrs exactly has num_lrs lr images
        """
        keys_to_features = {'lrs':tf.io.FixedLenFeature((self.num_lrs, 128, 128), tf.float32),
                            'hr': tf.io.FixedLenFeature((384, 384), tf.float32)}
        
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        
        lrs_merged = parsed_features['lrs']
        lrs_reshaped = tf.transpose(lrs_merged, perm=[1,2,0])
        
        if self.augment:
            return self.augment_data(lrs_reshaped, tf.expand_dims(parsed_features['hr'], axis=2))
        else:
             return lrs_reshaped, tf.expand_dims(parsed_features['hr'], axis=2)
        
        
        
    @tf.function
    def augment_data(self, lrs, hr):
        """ Randomly flip horizonally or vertically the input tensors
        
        """
        
        if tf.random.uniform(()) >= 0.5:
            lrs = tf.image.flip_left_right(lrs)
            hr = tf.image.flip_left_right(hr)
            
        if tf.random.uniform(()) >= 0.5:
            lrs = tf.image.flip_up_down(lrs)
            hr = tf.image.flip_up_down(hr)
        
        return lrs, hr
            