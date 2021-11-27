import tensorflow as tf
import skimage.io as io
import numpy as np


class DataSetCreator:
    def __init__(self, list_images_dir, list_labels_dir, batch_size, sample_weights=None, to_fit=True):
        # self.dataset = dataset
        self.list_images_dir = list_images_dir
        self.list_labels_dir = np.array(list(map(lambda x: list_labels_dir[x], self.list_images_dir)))
        self.dataset = tf.data.Dataset.from_tensor_slices((self.list_images_dir, self.list_labels_dir))
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        self.to_fit = to_fit

    @staticmethod
    def _load_image(path):
        path = path.numpy().decode("utf-8")
        image = io.imread(path, plugin='simpleitk')
        return image

    @staticmethod
    def _load_label(path):
        path = path.numpy().decode("utf-8")
        label = io.imread(path, plugin='simpleitk')
        return label

    def _create_sample_weights(self, label):

        # The weights for each class, with the constraint that:
        #     sum(class_weights) == 1.0
        class_weights = tf.constant(self.sample_weights)
        class_weights = class_weights / tf.reduce_sum(class_weights)

        # Create an image of `sample_weights` by using the label at each pixel as an
        # index into the `class weights` .
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
        return sample_weights

    def _load_data(self, x_path, y_path):
        image = self._load_image(x_path)
        label = self._load_label(y_path)
        if self.to_fit:
            sample_weights = self._create_sample_weights(label)
            return image, label, sample_weights
        else:
            data_id = [x_path, y_path]
            return image, label, data_id

    def _parse_function(self, x, y):
        if self.to_fit:
            return tf.py_function(self._load_data, (x, y), (tf.float32, tf.float32, tf.float64))
        else:
            return tf.py_function(self._load_data, (x, y), (tf.float32, tf.float32, tf.string))

    def load_process(self):
        self.loaded_dataset = self.dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.loaded_dataset = self.loaded_dataset.cache()

        # Create batches
        # self.loaded_dataset = self.loaded_dataset.repeat()
        # self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return self.loaded_dataset
