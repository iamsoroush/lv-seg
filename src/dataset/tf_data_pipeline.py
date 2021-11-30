import tensorflow as tf
import skimage.io as io
import numpy as np


class DataSetCreator:
    """
    How To:
    To create such a dataset we need to follow the below instructions:

    dataset_creator = DataSetCreator(x_train_dir, y_train_dir, sample_weights)
    train_data_gen = dataset_creator.load_process()
    """
    def __init__(self, list_images_dir, list_labels_dir, sample_weights=None, to_fit=True):
        """
        :param list_images_dir: list of images director
        :param list_labels_dir: list of labels directory
        :param sample_weights: sample weights for each instance
        :param to_fit: weather it is the test set od train/validation set
        """
        # self.dataset = dataset
        self.list_images_dir = list_images_dir
        self.list_labels_dir = np.array(list(map(lambda x: list_labels_dir[x], self.list_images_dir)))
        self.dataset = tf.data.Dataset.from_tensor_slices((self.list_images_dir, self.list_labels_dir))
        self.sample_weights = sample_weights
        self.to_fit = to_fit

    @staticmethod
    def _load_image(path):
        """
        loads the image form dataset directory
        :param path: image directory, str
        :return: loaded image, numpy.array
        """
        path = path.numpy().decode("utf-8")
        image = io.imread(path, plugin='simpleitk')
        return image

    @staticmethod
    def _load_label(path):
        """
        loads the label from dataset directory
        :param path: label directory, str
        :return: loaded image, numpy.array
        """
        path = path.numpy().decode("utf-8")
        label = io.imread(path, plugin='simpleitk')
        return label

    def _create_sample_weights(self, label):
        """
        create sample weights for the label and class_weights given
        :param label: input label, numpy.array
        :return: sample_weights, tf.int32
        """

        # The weights for each class, with the constraint that:
        #     sum(class_weights) == 1.0
        class_weights = tf.constant(self.sample_weights)
        class_weights = class_weights / tf.reduce_sum(class_weights)

        # Create an image of `sample_weights` by using the label at each pixel as an
        # index into the `class weights` .
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
        return sample_weights

    def _load_data(self, x_path, y_path):
        """
        integrates the images and labels loading and sample_weights
        :param x_path: images directory, str
        :param y_path: labels directory, str
        :return: a tuple of (image, label, sample_weights), or (image, label, data_id)
        """
        image = self._load_image(x_path)
        label = self._load_label(y_path)
        if self.to_fit:
            sample_weights = self._create_sample_weights(label)
            return image, label, sample_weights
        else:
            data_id = [x_path, y_path]
            return image, label, data_id

    def _parse_function(self, x, y):
        """
        returns the parse function to config for tf.data.Dataset
        :param x: images directory, list
        :param y: labels directory, list
        :return: tf.py_function
        """
        if self.to_fit:
            return tf.py_function(self._load_data, (x, y), (tf.float32, tf.float32, tf.float64))
        else:
            return tf.py_function(self._load_data, (x, y), (tf.float32, tf.float32, tf.string))

    def load_process(self):
        """
        loads the dataset and config the dataset by the parse function created beforehand and makes it parallel
        :return: loaded_dataset, tf.data.Dataset
        """
        self.loaded_dataset = self.dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.loaded_dataset = self.loaded_dataset.cache()

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return self.loaded_dataset
