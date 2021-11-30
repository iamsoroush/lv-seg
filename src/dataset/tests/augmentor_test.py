import os
from ..augmentor import Augmentor
import pytest
from ..dataset import DataLoader
from ..dataset_generator import DatasetGenerator
from abstractions.utils import load_config_file


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
        dataset = DataLoader(config, data_dir='/content/EchoNet-Dynamic')
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
    def augmentor(self, config):
        augmentor = Augmentor(config)
        return augmentor

    def test_add_augmentation(self, augmentor, data_gen):
        augmented_gen = augmentor.add_augmentation(data_gen)
        augmented_batch = next(augmented_gen)

        assert len(next(data_gen)) == 3
        assert len(next(data_gen)) == 3
        # Size checking
        assert augmented_batch[0].shape == (112, 112, 1)

        # Type checking
        assert 'float' in str(augmented_batch[0][0].dtype)
        assert 'generator' in str(type(augmented_gen))
