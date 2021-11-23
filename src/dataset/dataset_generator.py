import skimage.io as io  # to read the .mhd and .raw data
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


class DatasetGenerator(tf.keras.utils.Sequence):

    """
    making data generators for both train_set and test_set

    Example:

        train_data_gen = DatasetGenerator(x_train_dir, y_train_dir, batch_size,
                                          input_size, n_channels, to_fit, shuffle, seed)
        val_data_gen = DatasetGenerator(x_val_dir, y_val_dir, batch_size,
                                        input_size, n_channels, to_fit, shuffle, seed)

    """

    def __init__(self,
                 list_images_dir,
                 list_labels_dir,
                 batch_size,
                 input_size,
                 n_channels,
                 channel_last=True,
                 to_fit=True,
                 shuffle=True,
                 seed=None):

        """
        Handles data generators

        :param list_images_dir: list of images directory, list
        :param list_labels_dir: segmentation labels directory as values with the list_images_dir as keys, dict
        :param batch_size: batch size, int
        :param input_size: input image resolution, (h, w)
        :param n_channels: number of channels, int
        :param to_fit: for predicting time, bool
        :param shuffle: if True the dataset will shuffle with random_state of seed, bool
        :param seed: seed, int
        :param self.batch_index: keeping the current batch index using in next function, int
        :param self.indexes: index list of our dataset directory
        :param self.x: dataset image generated from list_images_dir by generate_X
        :param self.y: image labels generated from list_labels_dir by generate_y
        // changing from "input_res" to "input_size"
        """

        self.list_images_dir = list_images_dir
        self.list_labels_dir = list_labels_dir
        self.indexes = np.arange(len(self.list_images_dir))
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_channels = n_channels
        self.channel_last = channel_last
        self.seed = seed
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.batch_index = 0

    def __getitem__(self, index):

        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate indexes of the batch
        first_index = index * self.batch_size

        if (index + 1) * self.batch_size > len(self.list_images_dir):
            last_index = len(self.list_images_dir)
        else:
            last_index = (index + 1) * self.batch_size

        # selected indexes
        indexes = self.indexes[first_index: last_index]

        batch_image_dir = [k for k in self.list_images_dir[indexes]]
        x = self.generate_x(batch_image_dir)

        # returning the data using the selected indexes
        if self.to_fit:
            y = self.generate_y(batch_image_dir)
            return x, y
        else:
            return x

    def on_epoch_end(self):

        """shuffles indexes after each epoch"""

        if self.shuffle:
            np.random.RandomState(None).shuffle(self.indexes)

    def __len__(self):

        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """

        return int(np.ceil(len(self.list_images_dir) / self.batch_size))

    def get_n_iter(self):
        return self.__len__()

    def __next__(self):

        """create a generator that iterate over the Sequence."""

        return self.next()

    def next(self):

        """
        Create iteration through batches of the generator
        :return: next batch, np.ndarray
        """

        index = next(self.flow_index())
        return self.__getitem__(index)

    def reset(self):

        """reset indexes for iteration"""

        self.batch_index = 0

    def flow_index(self):

        """:yield: indexes for next function to iterate through indexes of data"""

        if len(self.list_images_dir) - 1 < self.batch_index * self.batch_size:
            self.reset()
        batch_index = self.batch_index
        self.batch_index += 1
        yield batch_index

    def generate_x(self, dir_list):
        """
        reads A4C view images from dataset

        :param dir_list:

        :return x_4_ch_preprocessed: array of preprocessed images
        """

        # reading images of .mhd format with the help of SimpleITK plugin,
        # and makes all of them channel last order.
        # X_4CH_ED: list[numpy.ndarray]

        x_4ch = list(map(lambda x: io.imread(x, plugin='simpleitk'),
                         dir_list))

        if len(x_4ch[0].shape) == 3:
            if self.channel_last:
                x_4ch = [np.moveaxis(x, 0, -1) for x in x_4ch]

        elif len(x_4ch[0].shape) == 2:
            if self.channel_last:
                x_4ch = [x[:, :, tf.newaxis] for x in x_4ch]
            else:
                x_4ch = [x[tf.newaxis, :, :] for x in x_4ch]

        # x_4ch_data = np.array(x_4ch, dtype=object)

        return x_4ch

    def generate_y(self, dir_list):

        """
        reads A4C view segmentation labels from dataset

        :param dir_list

        :return y_4_ch_preprocessed: array of preprocessed images
        """

        # reading segmentation labels of .mhd format with the help of SimpleITK plugin,
        # and makes all of them channel last order.
        # y_4ch: list[numpy.ndarray]

        y_4ch = list(map(lambda x: io.imread(x, plugin='simpleitk'),
                         [self.list_labels_dir[image_path] for image_path in dir_list]))

        if len(y_4ch[0].shape) == 3:
            if self.channel_last:
                y_4ch = [np.moveaxis(y, 0, -1) for y in y_4ch]

        # to categorical
        y_4ch_cat = [to_categorical(y) for y in y_4ch]

        # extract just left ventricle label from y_4ch_cat
        y_4ch_lv = [y[:, :, 1] for y in y_4ch_cat]

        # y_4ch_data = np.array(y_4ch_lv, dtype=object)

        return y_4ch_lv

    def random_visualization(self):

        """
        random visualization of an image
        """

        # choosing random image from dataset random indexes
        random_batch_index = np.random.randint(self.__len__())
        random_batch = self.__getitem__(random_batch_index)
        random_image_index = np.random.randint(len(random_batch[0]))
        random_image = random_batch[0][random_image_index]
        image_label = random_batch[1][random_image_index]

        print("image_shape: ", random_image.shape)
        print("label_shape: ", image_label.shape)
        self.visualization(random_image, image_label)

    @staticmethod
    def visualization(image, label):
        # setting a two-frame-image to plotting both the image and its segmentation labels
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        ax[0].imshow(image, cmap='gray')
        ax[0].axis('off')
        ax[1].imshow(label, cmap='gray')
        ax[1].axis('off')
        plt.show()
