import os
import sys
import tensorflow as tf
import pytest
sys.path.append(os.path.abspath('../..'))
from dataset.preprocessing_tf import PreprocessorTF
from dataset.data_loader_tf import EchoNetDataLoader
from utils import load_config_file


class TestClass:

    @pytest.fixture
    def config(self):
        root_dir = os.path.abspath(os.curdir)
        if 'lv-seg' not in root_dir:
            root_dir = os.path.join(root_dir, 'lv-seg').replace('\\', '/')
        config_path = os.path.join(root_dir, "./runs/template/config.yaml")
        print(config_path)
        config = load_config_file(config_path)
        return config

    @pytest.fixture
    def dataset(self, config):
        data_dir = config.data_loader.dataset_dir
        dataset = EchoNetDataLoader(data_dir, config)
        return dataset

    def test_add_image_preprocess(self, dataset, config):
        preprocessor = PreprocessorTF(config)
        preprocessed_image = preprocessor.add_image_preprocess(dataset)

        assert "MapDataset" in  str(type(preprocessed_image))

    def test_add_label_preprocess(self, dataset, config):
        preprocessor = PreprocessorTF(config)
        preprocessed_label = preprocessor.add_label_preprocess(dataset)

        assert "MapDataset" in str(type(preprocessed_label))

    def test_batchify(self, config, dataset):
        preprocessor = PreprocessorTF(config)

        generator_batch, n_iter = preprocessor.batchify(dataset, len(dataset))

        assert type(n_iter) == int and "RepeatDataset" in str(type(generator_batch))

    def test_add_preprocess(self, config, dataset):
        preprocessor = PreprocessorTF(config)
        gen = preprocessor.add_image_preprocess(dataset)
        gen = preprocessor.add_label_preprocess(gen)
        gen, n_iter = preprocessor.batchify(gen, len(dataset))

        assert type(n_iter) == int and "RepeatDataset" in str(type(gen))

    def test_rescale(self):
        image = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float32)
        min_val = 0
        max_val = 255
        rescaled_image = (image - min_val) / (max_val - min_val)

        assert 0 <= float(tf.reduce_max(rescaled_image)) <= 1 and 0 <= float(tf.reduce_min(rescaled_image)) <= 1

    def test_resize(self):
        image = tf.zeros([112, 112, 1])
        input_h = 128
        input_w = 128
        image_resized = tf.image.resize(image, [input_h, input_w],
                                        antialias=False,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        assert image_resized.shape == [input_h, input_w, 1]

    def test_convert_to_gray(self):
        image = tf.zeros([112, 112, 3])

        gray_image = tf.image.rgb_to_grayscale(image)

        assert gray_image.shape[-1] == 1
