# from abstractions import PreprocessorBase
# from abstractions.utils import ConfigStruct
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath('../../src'))
print(os.path.abspath('../../src'))
# sys.path.append(os.path.abspath('../lv_seg/'))
from dataset.data_loader import EchoNetDataLoader
from utils import load_config_file
from abstractions.preprocessing import PreprocessorBase
import matplotlib.pyplot as plt


class PreprocessorTF(PreprocessorBase):

    def image_preprocess(self, image):

        pre_processed_img = tf.reshape(image, shape=(112, 112, 1))
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

    def label_preprocess(self, label, weight):
        pre_processed_label = tf.reshape(label, shape=(112, 112))
        pre_processed_weight = tf.reshape(weight, shape=(112, 112))
        if self.do_resizing:
            pre_processed_label = self._resize(pre_processed_label[:, :, tf.newaxis])
            pre_processed_weight = self._resize(pre_processed_weight[:, :, tf.newaxis])

        return pre_processed_label, pre_processed_weight

    def _wrapper_image_preprocess(self, x, y, w):
        pre_processed = self.image_preprocess(x)
        return pre_processed, y, w

    def _wrapper_label_preprocess(self, x, y, w):
        pre_processed_y, pre_processed_w = self.label_preprocess(y, w)
        return x, pre_processed_y, pre_processed_w

    def add_image_preprocess(self, generator):
        return generator.map(self._wrapper_image_preprocess)

    def add_label_preprocess(self, generator):
        return generator.map(self._wrapper_label_preprocess)

    def batchify(self, generator, n_data_points):
        generator = generator.batch(self.batch_size).repeat()
        n_iter = n_data_points // self.batch_size + int((n_data_points % self.batch_size) > 0)

        return generator, n_iter

    def _resize(self, image):
        image_resized = tf.image.resize(image, [self.input_h, self.input_w],
                                        antialias=False,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image_resized

    @staticmethod
    def _rescale(image, min_val, max_val):

        rescaled_image = (image - min_val) / (max_val - min_val)

        return rescaled_image

    @staticmethod
    def _convert_to_gray(image):
        gray_image = tf.image.rgb_to_grayscale(image)
        return gray_image

    def add_preprocess(self, generator, n_data_points):
        gen = self.add_image_preprocess(generator)
        gen = self.add_label_preprocess(gen)
        gen, n_iter = self.batchify(gen, n_data_points)
        return gen, n_iter

    def _load_params(self, config):
        # self.normalize_by = config.preprocessor.normalize_by
        self.input_h = config.input_height
        self.input_w = config.input_width
        self.batch_size = config.batch_size

        self.do_resizing = config.preprocessor.do_resizing
        self.do_normalization = config.preprocessor.do_normalization
        self.min = config.preprocessor.min
        self.max = config.preprocessor.max

    def _set_defaults(self):

        self.normalize_by = 255
        self.input_h = 128
        self.input_w = 128
        self.batch_size = 8

        self.do_resizing = True
        self.do_normalization = True
        self.min = 0
        self.max = 255


if __name__ == '__main__':
    root_dir = os.path.abspath(os.curdir)
    if 'lv-seg' not in root_dir:
        root_dir = os.path.join(root_dir, 'lv-seg').replace('\\', '/')
    config_path = os.path.join(root_dir, "../../runs/template/config.yaml")
    config = load_config_file(config_path)

    dataset_obj = EchoNetDataLoader(config)
    train_gen, train_n_iter = dataset_obj.create_train_data_generator()
    val_gen, val_n_iter = dataset_obj.create_validation_data_generator()
    test_gen, test_n_iter = dataset_obj.create_test_data_generator()
    print(train_n_iter)

    pre_processor = PreprocessorTF()
    pre_processed_dataset = pre_processor.add_preprocess(train_gen, train_n_iter)

    for i, ele in zip(range(0, 1), pre_processed_dataset):
        print(i)
        print(len(ele[0]))
        print(ele[0].numpy().shape)
        first_img = ele[0][0]
        img_label = ele[1][0]
        img_weights = ele[2][0]
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(first_img)
        ax[1].imshow(img_label)
        ax[2].imshow(img_weights)
        plt.show()
