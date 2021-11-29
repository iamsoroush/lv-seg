import os
import sys
sys.path.append(os.path.abspath('../..'))
print(os.path.abspath('../..'))
# sys.path.append(os.path.abspath('../lv_seg/'))
from dataset.data_loader import EchoNetDataLoader
from utils import load_config_file
# from echotrain.utils import load_config_file
import pytest
import matplotlib.pyplot as plt


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

    @pytest.mark.parametrize("x, y", [
        ([1, 2, 3], {1: 4, 2: 5, 3: 6}),
    ])
    def test_shuffle_func(self, dataset, x, y):
        shuffled_x, shuffled_y = dataset._shuffle_func(x, y)

        # Testing if both data and labels are shuffled the same.
        assert shuffled_x == list(shuffled_y.keys())

    def test_create_training_generator(self, dataset):
        train_gen, train_n = dataset.create_training_generator()

        # Testing if the number of iterations are equal to the real numbers as in the echonet dataset
        assert train_n == len(train_gen)
        # assert n_iter_val == len(dataset.x_val_dir) / dataset.batch_size

    def test_create_validation_generator(self, dataset):
        val_gen, val_n = dataset.create_validation_generator()

        # Testing if the number of iterations are equal to the real numbers as in the echonet dataset
        assert val_n == len(val_gen)
        # assert n_iter_val == len(dataset.x_val_dir) / dataset.batch_size

    def test_create_test_generator(self, dataset):
        test_gen, test_n = dataset.create_test_generator()

        # Testing if the number of iterations are equal to the real numbers as in the echonet dataset
        assert test_n == len(test_gen)

# if __name__ == '__main__':
#     root_dir = os.path.abspath(os.curdir)
#     if 'lv-seg' not in root_dir:
#         root_dir = os.path.join(root_dir, 'lv-seg').replace('\\', '/')
#     config_path = os.path.join(root_dir, "../../../runs/template/config.yaml")
#     config = load_config_file(config_path)
#
#     dataset_obj = EchoNetDataLoader(config)
#     train_gen, train_n_iter = dataset_obj.create_train_data_generator()
#     val_gen, val_n_iter = dataset_obj.create_validation_data_generator()
#     test_gen, test_n_iter = dataset_obj.create_test_data_generator()
#     print(train_n_iter)
#
#     for i, ele in zip(range(0, 1), train_gen):
#         print(i)
#         print(len(ele[0]))
#         print(ele[0].numpy().shape)
#         first_img = ele[0][0]
#         img_label = ele[1][0]
#         img_weights = ele[2][0]
#         fig, ax = plt.subplots(1, 3)
#         ax[0].imshow(first_img)
#         ax[1].imshow(img_label)
#         ax[2].imshow(img_weights)
#         plt.show()
#
#     for i, ele in zip(range(0, 1), val_gen):
#         print(i)
#         print(len(ele[0]))
#         print(ele[0].numpy().shape)
#         first_img = ele[0][0]
#         img_label = ele[1][0]
#         img_weights = ele[2][0]
#         fig, ax = plt.subplots(1, 3)
#         ax[0].imshow(first_img)
#         ax[1].imshow(img_label)
#         ax[2].imshow(img_weights)
#         plt.show()
#
#     # list_dataset = list(train_gen.as_numpy_iterator())
#     for i, ele in zip(range(0, 1), test_gen):
#         print(i)
#         print(len(ele[0]))
#         print(ele[0].numpy().shape)
#         first_img = ele[0][0]
#         img_label = ele[1][0]
#         img_id = ele[2][0]
#         fig, ax = plt.subplots(1, 2)
#         ax[0].imshow(first_img)
#         ax[1].imshow(img_label)
#         plt.show()
#         print(img_id)
