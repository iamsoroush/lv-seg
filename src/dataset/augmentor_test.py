import os
import pytest
# from echotrain.dataset.dataset_echonet import EchoNetDataset
# from echotrain.dataset.dataset_generator import DatasetGenerator
# from echotrain.model.augmentation import Augmentation
# from echotrain.utils import load_config_file
from .augmentor import Augmentor
import pytest
from .dataset import DataLoader
from abstractions import DataLoaderBase
from .dataset_generator import DatasetGenerator
import random

import yaml
import pathlib

import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name, base_dir: pathlib.Path):
    """Sets up mlflow and returns an ``active_run`` object.

    tracking_uri/
        experiment_id/
            run1
            run2
            ...

    :param mlflow_tracking_uri: ``tracking_uri`` for mlflow
    :param mlflow_experiment_name: ``experiment_name`` for mlflow, use the same ``experiment_name`` for all experiments
     related to the same task. This is different from the ``experiment`` concept that we use.
    :param base_dir: directory for your experiment, containing your `config.yaml` file.

    :returns active_run: an ``active_run`` object to use for mlflow logging.

    """

    # Loads run_id if exists
    run_id_path = base_dir.joinpath('run_id.txt')
    run_name = base_dir.name

    if run_id_path.exists():
        with open(run_id_path, 'r') as f:
            run_id = f.readline()
    else:
        run_id = None

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient(mlflow_tracking_uri)

    # Create new run if run_id does not exist
    if run_id is not None:
        mlflow.set_experiment(mlflow_experiment_name)
        active_run = mlflow.start_run(run_id=run_id)
    else:
        experiment = client.get_experiment_by_name(mlflow_experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(mlflow_experiment_name)

        active_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

    return active_run


def check_for_config_file(experiment_dir: pathlib.Path):
    """Checks for existence of config file and returns the path to config file if exists."""

    if not experiment_dir.is_dir():
        raise Exception(f'{experiment_dir} is not a directory.')

    yaml_files = list(experiment_dir.glob('*.yaml'))
    if not any(yaml_files):
        raise Exception(f'no .yaml files found.')
    elif len(yaml_files) > 1:
        raise Exception(f'found more than one .yaml files.')

    return yaml_files[0]


def load_config_file(path):
    """
    loads the json config file and returns a dictionary

    :param path: path to json config file
    :return: a dictionary of {config_name: config_value}
    """

    with open(path) as f:
        data_map = yaml.safe_load(f)

    config_obj = Struct(**data_map)
    return config_obj


class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v


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

    def test_batch_augmentation(self, augmentor, data_gen):
        batch = data_gen.next()
        augmented_batch = augmentor.batch_augmentation(batch)

        assert len(augmented_batch[0]) == data_gen.batch_size

        assert augmented_batch[0][0].shape == (112, 112, 1)

        # Type checking
        assert 'float' in str(augmented_batch[0][0].dtype)

    def test_add_augmentation(self, augmentor, data_gen):
        augmented_gen = augmentor.add_augmentation(data_gen)

        assert 'generator' in str(type(augmented_gen))

        augmented_batch = next(augmented_gen)

        assert augmented_batch[0][0].shape == (112, 112, 1)

        # Type checking
        assert 'float' in str(augmented_batch[0][0].dtype)
