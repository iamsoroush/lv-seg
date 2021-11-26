import os
import sys
sys.path.append(os.path.abspath('../..'))
print(os.path.abspath('../..'))
# sys.path.append(os.path.abspath('../lv_seg/'))
from dataset.data_loader import EchoNetDataLoader
from utils import load_config_file
from echotrain.utils import load_config_file
import pytest
import matplotlib.pyplot as plt


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
        dataset = EchoNetDataLoader(config)
        return dataset

    @pytest.mark.parametrize("x, y", [
        ([1, 2, 3], {1: 4, 2: 5, 3: 6}),
    ])
    def test_shuffle_func(self, dataset, x, y):
        shuffled_x, shuffled_y = dataset._shuffle_func(x, y)

        # Testing if both data and labels are shuffled the same.
        assert shuffled_x == list(shuffled_y.keys())

    def test_create_data_generators(self, dataset):
        train_gen, val_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

        # Testing if the number of iterations are equal to the real numbers as in the echonet dataset
        assert n_iter_train == len(dataset.x_train_dir) / dataset.batch_size
        assert n_iter_val == len(dataset.x_val_dir) / dataset.batch_size

    def test_create_test_data_generator(self, dataset):
        test_gen, n_iter_test = dataset.create_test_data_generator()

        # Testing if the number of iterations are equal to the real numbers as in the echonet dataset
        assert n_iter_test == len(dataset.x_test_dir) / dataset.batch_size


if __name__ == '__main__':
    root_dir = os.path.abspath(os.curdir)
    if 'lv-seg' not in root_dir:
        root_dir = os.path.join(root_dir, 'lv-seg').replace('\\', '/')
    config_path = os.path.join(root_dir, "../../../runs/template/config.yaml")
    config = load_config_file(config_path)

    dataset_obj = EchoNetDataLoader(config)
    train_gen, n_iter = dataset_obj.create_train_data_generator()
    print(n_iter)

    # list_dataset = list(train_gen.as_numpy_iterator())
    for i, ele in zip(range(0, 2), train_gen):
        print(i)
        print(len(ele[0]))
        print(ele[0].numpy().shape)
        first_img = ele[0][0]
        img_label = ele[1][0]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(first_img)
        ax[1].imshow(img_label)
        plt.show()
        # print(len(ele[1]))
