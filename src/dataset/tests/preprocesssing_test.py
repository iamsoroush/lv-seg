from ..preprocessing import Preprocessor
import os
from ..dataset_generator import DatasetGenerator
import pytest
from ..dataset import DataLoader
from ...utils import *
import types


class TestClass:

    @pytest.fixture
    def config(self):
        root_dir = os.path.abspath(os.curdir)
        if 'lv-seg' not in root_dir:
            root_dir = os.path.join(root_dir, 'lv-seg').replace('\\', '/')
        config_path = os.path.join(root_dir, "runs/template/config.yaml")
        config = load_config_file(config_path)
        return config

    @pytest.fixture
    def dataset(self, config):
        dataset = DataLoader(config, '/content/EchoNet-Dynamic')
        return dataset

    @pytest.fixture
    def generator_inputs(self, dataset):
        instance = {
            'list_images_dir': dataset.x_train_dir,
            'list_labels_dir': dataset.y_train_dir,
            'batch_size': dataset.batch_size,
            'input_size': dataset.input_size,
            'n_channels': dataset.n_channels,
            'channel_last': True,
            'to_fit': dataset.to_fit,
            'shuffle': dataset.shuffle,
            'seed': dataset.seed
        }
        return instance

    @pytest.fixture
    def data_gen(self, generator_inputs):
        data_gen = DatasetGenerator(**generator_inputs)
        return data_gen

    @pytest.fixture
    def pre_processor(self, config):
        pre_processor = Preprocessor(config)
        return pre_processor

    def test_image_preprocess(self, pre_processor, data_gen):
        img = next(data_gen)[0]
        pre_processed_img = pre_processor.image_preprocess(img)

        assert 'numpy.ndarray' in str(type(img))

        assert img.shape == (112, 112, 1)

        # Resizing
        assert pre_processed_img.shape == (data_gen.input_size[0], data_gen.input_size[1], 1)

        # Normalization
        assert 0 <= pre_processed_img.all() <= 1

        # Type checking
        assert 'float' in str(pre_processed_img.dtype)

    def test_label_preprocess(self, pre_processor, data_gen):
        label_img = next(data_gen)[1]

        assert 'numpy.ndarray' in str(type(label_img))

        assert 'float' in str(label_img.dtype)

        assert label_img.shape == (112, 112)

    def test_add_image_preprocess(self, pre_processor, data_gen):
        pre_processed_batch = pre_processor.add_image_preprocess(data_gen)

        # Size checking
        assert next(pre_processed_batch)[0].shape == (128, 128, 1)

        # Type checking
        assert 'float' in str(next(pre_processed_batch)[0].dtype)

    def test_add_label_preprocess(self, pre_processor, data_gen):
        pre_processed_batch = pre_processor.add_label_preprocess(data_gen)

        # Size checking
        assert next(pre_processed_batch)[1].shape == (128, 128, 1)

        # Type checking
        assert 'float' in str(next(pre_processed_batch)[1].dtype)

    def test_batchify(self, pre_processor, data_gen, dataset):
        train_gen, n_train = dataset.create_training_generator()
        n_data_points = n_train
        image_batchify = pre_processor.batchify(data_gen, n_data_points)
        assert 'int' in str(type(image_batchify[1]))
        assert 'int' in str(type(n_data_points))
        assert len(next(image_batchify[0])) == 3

    def test_add_preprocess(self, pre_processor, data_gen, dataset):
        train_gen, n_train = dataset.create_training_generator()
        n_data_points = n_train
        pre_processed_image = pre_processor.add_preprocess(data_gen, n_data_points)
        assert 'int' in str(type(pre_processed_image[1]))
        assert 'int' in str(type(n_data_points))
        assert len(next(pre_processed_image[0])) == 3
