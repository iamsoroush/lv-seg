from abstractions import PreprocessorBase
import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray


class Preprocessor(PreprocessorBase):
    """
     PreProcessing module used for images, batches, generators

    Example:

        preprocessor = PreProcess()
        image = preprocessor.img_preprocess(image)
        X, y = preprocessor.batch_preprocess(gen_batch)
        data_gen = preprocessor.add_preprocess(data_gen, add_augmentation=True)

    Attributes:

        target_size: image target size for resizing, tuple (image_height, image_width)
        min: minimum value of the image range, int
        max: maximum value of the image range, int
        normalization: for rescaling the image, bool

    """

    def image_preprocess(self, image):

        """
        pre-processing on input image
        Args:
            image: input image, np.array


        Returns:
            pre_processed_img
        """

        pre_processed_img = image.copy()

        # converting the images to grayscale
        if len(image.shape) != 2 and image.shape[-1] != 1:
            pre_processed_img = self._convert_to_gray(pre_processed_img)

        # resizing
        if self.do_resizing:
            pre_processed_img = self._resize(pre_processed_img)

        # normalization on the given image
        if self.do_normalization:
            pre_processed_img = self._rescale(pre_processed_img, self.min, self.max)

        return pre_processed_img

    def label_preprocess(self, label):

        """
        pre-processing on input label
        Args:
            label: input label, np.array

        Returns:
            pre-processed label
        """

        if self.do_resizing:
            label = self._resize(label[:, :, tf.newaxis])

        return label

    def batch_preprocess(self, batch):

        """
        batch pre_processing function
        Args:
            batch: input batch (X, y)

        Returns:
            tuple(x_preprocessed_batch, y_preprocessed_batch):
            - x_preprocessed_batch: preprocessed batch for x

            - y_preprocessed_batch: preprocessed batch for y
        """

        # images of the give batch
        x = batch[0]

        # labels of the give batch
        y = batch[1]

        # pre-processing every image of the batch given
        x_preprocessed_batch = np.array(list(self.image_preprocess(x)))
        # the labels of the batches do not need pre-processing (yet!)
        y_preprocessed_batch = np.array(list(self.label_preprocess(y)))

        return x_preprocessed_batch, y_preprocessed_batch

    def add_image_preprocess(self, generator):
        """
        Plugs input-image-preprocessing on top of the given generator
        Args:
            generator: a `Python generator` which yields a single data-point ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator`` in which

                - ``x`` => input image,
                - ``y`` => label, or segmentation map for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)

        Returns:
            A ``generator` with preprocessed ``x`` s

        """

        while True:
            batch = next(generator)
            label = batch[1]
            third_element = batch[2]
            pre_processed_batch = self.batch_preprocess(batch)
            pre_processed_batch = (pre_processed_batch[0], label, third_element)
            yield pre_processed_batch

    def add_label_preprocess(self, generator):
        """
        Plugs input-label-preprocessing on top of the given ``generator``
        Args:
            generator: a ``Python generator`` which yields a single data-point ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator`` in which

                - ``x`` => input image,
                - ``y`` => label, or segmentation map for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)

        Returns:
            A ``generator`` with preprocessed ``y`` s

        """
        while True:
            batch = next(generator)
            image = batch[0]
            third_element = batch[2]
            pre_processed_batch = self.batch_preprocess(batch)
            pre_processed_batch = (image, pre_processed_batch[1], third_element)
            yield pre_processed_batch

    def batchify(self, generator, n_data_points):
        """
        Batchifies the given ``generator``
        Args:
            generator: a ``Python generator`` which yields a single data-point ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator`` in which

                - ``x`` => input image,
                - ``y`` => label, or segmentation map for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)
            n_data_points: number of data_points in this sub-set.

        Returns:
            tuple(batched_generator, n_iter):
            - batched_generator: A repeated ``generator`` which yields a batch of data for each iteration, ``(x_batch, y_batch, sample_weights)`` or ``(x_batch, y_batch, data_ids)`` for test data gen, in which

                - ``x`` => (batch_size, input_h, input_w, n_channels)
                - ``y`` => classification: (batch_size, n_classes[or 1]), segmentation: (batch_size, input_h, input_w, n_classes)
                - ``sample_weights`` => (batch_size, 1)(classification/segmentation), (batch_size, input_h, input_w, 1)(segmentation)

            - n_iter: number of iterations per epoch

        """
        n_iter = n_data_points // self.batch_size + int((n_data_points % self.batch_size) > 0)
        gen = self._batch_gen(generator, self.batch_size)
        return gen, n_iter

    @staticmethod
    def _batch_gen(generator, batch_size):
        """
        Helps to batchify the given ``generator``
        Args:
            generator: a ``Python generator`` which yields a single data-point ``(x, y, sample_weight)`` or ``(x, y, data_id)`` if it is ``test_data_generator`` in which

                - ``x`` => input image,
                - ``y`` => label, or segmentation map for segmentation
                - ``sample_weight`` => float (classification/segmentation), or one-channel segmentation map (segmentation)
            batch_size: size of the ``generator`` batch

        Returns:
            tuple(x, y, input_w, sample_weight):
                - ``x`` => (batch_size, input_h, input_w, n_channels)
                - ``y`` => classification: (batch_size, n_classes[or 1]), segmentation: (batch_size, input_h, input_w, n_classes)
                - ``sample_weights`` => (batch_size, 1)(classification/segmentation), (batch_size, input_h, input_w, 1)(segmentation)

        """
        while True:
            x_b, y_b, z_b = list(), list(), list()
            for i in range(batch_size):
                x, y, z = next(generator)
                x_b.append(x)
                y_b.append(y)
                z_b.append(z)
            yield np.array(x_b), np.array(y_b), np.array(z_b)

    def add_preprocess(self, generator, n_data_points):

        """
        providing the suggested pre-processing for the given generator
        Args:
            generator: input generator ready for pre-processing, data generator < class DataGenerator >
            n_data_points: number of data_points in this sub-set.

        Returns:
            tuple(batched_generator, n_iter):
            - batched_generator: A repeated ``generator`` which yields a batch of data for each iteration, ``(x_batch, y_batch, sample_weights)`` or ``(x_batch, y_batch, data_ids)`` for test data gen, in which

                - ``x`` => (batch_size, input_h, input_w, n_channels)
                - ``y`` => classification: (batch_size, n_classes[or 1]), segmentation: (batch_size, input_h, input_w, n_classes)
                - ``sample_weights`` => (batch_size, 1)(classification/segmentation), (batch_size, input_h, input_w, 1)(segmentation)

            - n_iter: number of iterations per epoch
        """

        generator = self.add_image_preprocess(generator)
        generator = self.add_label_preprocess(generator)
        generator, n_iter = self.batchify(generator, n_data_points)

        return generator, n_iter

    def _load_params(self, config):

        """
        Read parameters from config file.
        Args:
            config: dict of a config file

        Returns:

        """
        self.input_h = config.input_height
        self.input_w = config.input_width
        self.max = config.preprocessor.max
        self.min = config.preprocessor.min
        self.do_resizing = config.preprocessor.do_resizing
        self.do_normalization = config.preprocessor.do_normalization
        self.batch_size = config.batch_size

    def _set_defaults(self):

        """
        Set default values for PreProcessing class
        Returns:

        """
        self.input_h = 128
        self.input_w = 128
        self.max = 255
        self.min = 0
        self.do_resizing = True
        self.do_normalization = True
        self.batch_size = 8

    @property
    def target_size(self):
        """
        Return the height and width of the input data

        Returns:
             tuple(input_h, input_w):
            - input_h: Height of the input data

            - input_w: Width of the input data
        """
        return self.input_h, self.input_w

    def _resize(self, image):

        """
        resizing image into the target_size dimensions
        Args:
            image: input image, np.array

        Returns:
            resized image

        """

        image_resized = np.array(tf.image.resize(image,
                                                 self.target_size,
                                                 antialias=False,
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        return image_resized

    @staticmethod
    def _rescale(image, min_val, max_val):

        """
        rescaling the input image
        Args:
            image: input image, np.array
            min_val: minimum value of the image
            max_val: maximum value of the image

        Returns:

        """

        rescaled_image = (image - min_val) / (max_val - min_val)

        return rescaled_image

    @staticmethod
    def _convert_to_gray(image):

        """
        converting the input image to grayscale, if needed
        Args:
            image: input image, np array

        Returns:
            converted image

        """

        gray_image = rgb2gray(image)
        return gray_image
