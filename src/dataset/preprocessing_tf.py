
import tensorflow as tf
from abstractions.preprocessing import PreprocessorBase


class PreprocessorTF(PreprocessorBase):

    def image_preprocess(self, image):
        """

        Args:
            image: tensor containing single image

        Returns:
            preprocessed image with type tensor

        in this method all preprocessing methods(converting ti gray , resizing , normalization)
        will be done  on input based on config  setting

        Notes : AT first the input tensor will be reshaped into its original shape to evoid  further errors
        THIS IS A MUST , and this is usable if only all your data in your dataset , have  the  same shape


        """

        pre_processed_img = tf.reshape(image, shape=(
            self.original_input_h, self.original_input_w, self.original_input_channel))
        # pre_processed_img = image

        # converting the images to grayscale
        if len(pre_processed_img.shape) != 2 and pre_processed_img.shape[-1] != 1:
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

        Args:
            label: a tensor containing single label


        Returns: preprocessed label tensor

        in this method resizing will be done based on config setting

        Notes : AT first the input tensors will be reshaped into their original shape to evoid  further errors
        THIS IS A MUST , and this is usable if only all your data in your dataset , have  the  same shape

        """
        pre_processed_label = tf.reshape(label, shape=(self.original_input_h, self.original_input_w))

        if self.do_resizing:
            pre_processed_label = self._resize(pre_processed_label[:, :, tf.newaxis])

        return pre_processed_label

    def weight_preprocess(self, weight):
        """

        Args:
            weight: a tensor containing single weight

        Returns: preprocessed weight tensor

        in this method resizing will be done based on config setting

        Notes : AT first the input tensors will be reshaped into their original shape to evoid  further errors
        THIS IS A MUST , and this is usable if only all your data in your dataset , have  the  same shape

        """
        pre_processed_weight = tf.reshape(weight, shape=(self.original_input_h, self.original_input_w))

        if self.do_resizing:
            pre_processed_weight = self._resize(pre_processed_weight[:, :, tf.newaxis])

        return pre_processed_weight

    def _wrapper_image_preprocess(self, x, y, w):
        """
        this method calls image_preprocess on x and do preprocessing just on x ,
        y and w will not change here

        Args:
            x: a tensor containing single image
            y: a tensor containing single label
            w: a tensor containing single weight

        Returns:
            pre_processed x ,  y , w
        """
        pre_processed = self.image_preprocess(x)
        return pre_processed, y, w

    def _wrapper_label_preprocess(self, x, y, w):
        """

        Args:
            x: a tensor containing single image
            y: a tensor containing single label
            w: a tensor containing single weight

        Returns:  x ,  pre_processed y , pre_processed w

        this method calls label_preprocess on y ,w  and do preprocessing just on y ,w ,
        x will not change here

        """
        pre_processed_y = self.label_preprocess(y)
        return x, pre_processed_y, w

    def _wrapper_weight_preprocess(self, x, y, w):
        preprocessed_w = self.weight_preprocess(w)
        return x, y, preprocessed_w

    def add_image_preprocess(self, generator):
        """

        Args:
            generator: input dataset ,  type = (tensor dataset)

        Returns:
            a tensor dataset with preprocessed x (images)

        in this method , the map method will be called on our tensor dataset , and only the images will become
        preprocessed

        """
        return generator.map(self._wrapper_image_preprocess)

    def add_label_preprocess(self, generator):
        """

         Args:
             generator: input dataset ,  type = (tensor dataset)

         Returns:
             a tensor dataset with preprocessed y and preprocessed w (label and weight)

         in this method , the map method will be called on our tensor dataset , and only the label and weight will
         become preprocessed

         """

        return generator.map(self._wrapper_label_preprocess)

    def add_weight_preprocess(self, generator):

        return generator.map(self._wrapper_weight_preprocess)

    def batchify(self, generator, n_data_points):
        """

        Args:
            generator:  a tensor dataset
            n_data_points: number of samples

        Returns: generator = a tensor dataset witch is batched now ,
                 n_iter  = type(int) number of steps per epoch

        """
        generator = generator.batch(self.batch_size).prefetch(2).repeat()
        n_iter = n_data_points // self.batch_size + int((n_data_points % self.batch_size) > 0)

        return generator, n_iter

    def _resize(self, image):
        """
        Args:
            image: input image , type  =tensor

        Returns:
            resized image ,  type  =tensor

        resizing image into the target_size dimensions

        """
        image_resized = tf.image.resize(image, [self.input_h, self.input_w],
                                        antialias=False,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image_resized

    @staticmethod
    def _rescale(image, min_val, max_val):
        """
        rescaling the input image
        Args:
            image: input image , tensor
            min_val: minimum value of the image
            max_val: maximum value of the image

        Returns:
            rescaled image

        """

        rescaled_image = (image - min_val) / (max_val - min_val)

        return rescaled_image

    @staticmethod
    def _convert_to_gray(image):
        """
        converting the input image to grayscale, if needed

        Args:
            image: input image , tensor

        Returns:
            converted image

        """
        gray_image = tf.image.rgb_to_grayscale(image)
        return gray_image

    def add_preprocess(self, generator, n_data_points):
        """
        Do preprocessing on all parts(image , label , weight) of out input tensor dataset
        and do batchify on our input dataset

        Args:
            generator: input dataset ,  type = (tensor dataset)
            n_data_points: number of samples

        Returns:
            gen : tensor dataset witch is batched and preprocessed
            n_iter : number of steps per  epoch

        """
        gen = self.add_image_preprocess(generator)
        gen = self.add_label_preprocess(gen)
        gen = self.add_weight_preprocess(gen)
        gen, n_iter = self.batchify(gen, n_data_points)
        return gen, n_iter

    def _load_params(self, config):

        self.input_h = config.input_height
        self.input_w = config.input_width
        self.batch_size = config.batch_size
        self.original_input_h = config.original_input_h
        self.original_input_w = config.original_input_w
        self.original_input_channel = config.original_input_channel

        self.do_resizing = config.preprocessor.do_resizing
        self.do_normalization = config.preprocessor.do_normalization
        self.min = config.preprocessor.min
        self.max = config.preprocessor.max

    def _set_defaults(self):

        self.input_h = 128
        self.input_w = 128
        self.batch_size = 32
        # for echonet
        self.original_input_h = 112
        self.original_input_w = 112
        self.original_input_channel = 1

        self.do_resizing = True
        self.do_normalization = True
        self.min = 0
        self.max = 255
