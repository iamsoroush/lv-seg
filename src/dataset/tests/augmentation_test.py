import os
import pytest
from ..data_loader import EchoNetDataLoader
from ..data_loader import DataSetCreator
from dataset.augmentation import Augmentor
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
        dataset, train_n = dataset.create_training_generator()
        return dataset

    @pytest.fixture
    def generator_inputs(self, dataset):
        instance = {
            'list_images_dir': dataset.x_train_dir,
            'list_labels_dir': dataset.y_train_dir,
            'batch_size': dataset.batch_size,
            'sample_weights': dataset.sample_weights,
            'to_fit': dataset.to_fit,
        }
        return instance

    # @pytest.fixture
    # def data(self, generator_inputs):
    #     data = DataSetCreator(**generator_inputs)
    #     return data

    @pytest.fixture
    def augmentor(self, config):
        augmentor = Augmentor(config)
        return augmentor

    # def test_batch_augmentation(self, augmentor, data_gen):
    #     batch = data_gen.next()
    #
    #     augmented_batch = augmentor.batch_augmentation(batch)
    #
    #     assert len(augmented_batch[0]) == data_gen.batch_size
    #
    #     assert augmented_batch[0][0].shape == (112, 112, 1)
    #
    #     # Type checking
    #     assert 'float' in str(augmented_batch[0][0].dtype)

    def test_add_augmentation(self, augmentor, dataset):
        augmented = augmentor.add_augmentation(dataset)

        assert 'tensor' in str(type(augmented))

        augmented_batch = next(iter(augmented))

        assert augmented_batch[0].shape == (112, 112)

        # Type checking
        assert 'float' in str(augmented_batch[0].dtype)
