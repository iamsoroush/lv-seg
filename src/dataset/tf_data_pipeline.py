import tensorflow as tf
import skimage.io as io
import numpy as np
import os


class DataSetCreator:
    def __init__(self, list_images_dir, list_labels_dir, batch_size):
        # self.dataset = dataset
        self.list_images_dir = list_images_dir
        self.list_labels_dir = np.array(list(map(lambda x: list_labels_dir[x], self.list_images_dir)))
        print(type(self.list_images_dir), type(self.list_labels_dir))
        print(self.list_images_dir.shape, self.list_labels_dir.shape)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.list_images_dir, self.list_labels_dir))
        self.batch_size = batch_size

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

    def _load_data(self, x_path, y_path):
        image = self._load_image(x_path)
        label = self._load_label(y_path)
        return image, label

    def _parse_function(self, x, y):
        return tf.py_function(self._load_data, (x, y), (tf.float32, tf.float32))

    def load_process(self):
        self.loaded_dataset = self.dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.loaded_dataset = self.loaded_dataset.cache()

        # Create batches
        self.loaded_dataset = self.loaded_dataset.repeat()
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return self.loaded_dataset

    def get_n_iter(self):
        return int(tf.math.ceil(len(self.list_images_dir) / self.batch_size))

    def get_batch(self):
        return next(iter(self.loaded_dataset))
