from abstractions import DataLoaderBase
from src.dataset.dataset_generator import DatasetGenerator
import os
from echotrain.dataset.dataset_echonet import EchoNetDataset
from echotrain.utils import load_config_file
import pytest
from src.dataset.dataset import DataLoader


class TestClass:

    @pytest.fixture
    def config(self):
        root_dir = os.path.abspath(os.curdir)
        if 'echotrain' not in root_dir:
            root_dir = os.path.join(root_dir, 'echotrain').replace('\\', '/')
        config_path = os.path.join(root_dir, "config/config_example_echonet.yaml")
        config = load_config_file(config_path)
        return config

    @pytest.fixture
    def dataset(self, config):
        dataset = EchoNetDataset(config)
        return dataset

    @pytest.mark.parametrize("x, y", [
        ([1, 2, 3], {1: 4, 2: 5, 3: 6}),
    ])
    def test_shuffle_func(self, dataset, x, y):
        shuffled_x, shuffled_y = dataset._shuffle_func(x, y)

        # Testing if both data and labels are shuffled the same.
        assert shuffled_x == list(shuffled_y.keys())

    def test_training_generator(self, dataset):
        train_gen, n_train = dataset.create_training_generator()

        assert n_train == len(dataset.x_train_dir) / dataset.batch_size

    def test_validation_generator(self, dataset):
        val_gen, val_n = dataset.create_validation_generator()

        assert val_n == len(dataset.x_val_dir) / dataset.batch_size

    def test_test_generator(self, dataset):
        test_gen, test_n = dataset.create_test_generator()

        assert test_n == len(dataset.x_test_dir) / dataset.batch_size
