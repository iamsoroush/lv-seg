from functools import partial
import os
import sys
import tensorflow as tf
from abstractions.augmentation import AugmentorBase
from tensorflow.python.data import AUTOTUNE
import albumentations as A
from albumentations import (
    Compose)


class Aug(AugmentorBase):
    """
    this class aims to implement augmentation on train and validation data

    HOW TO:

    augmentation = Aug(config)
    train_data_augmented = augmentation.add_augmentation(train_dataset)
    val_data_augmented = augmentation.add_augmentation(val_dataset)
    """

    def _set_defaults(self):
        """
        this method use the default values for augmentation in case the input of Aug class is None
        """
        self.do_aug_train = True
        self.do_aug_val = False
        self.rotation_proba = 0.5
        self.rotation_range = 20
        self.flip = 0.5

    def _load_params(self, config):
        """
        this method uses the parameters in config file for augmentation
        :param config: config file
        """
        config = config
        self.do_aug_train = config.do_train_augmentation
        self.do_aug_val = config.do_validation_augmentation
        self.rotation_proba = config.augmentator.rotation_proba
        self.rotation_range = config.augmentator.rotation_range
        self.flip = config.augmentator.flip_proba

    def aug_fn(self, image, label, weight):
        """
        this method is the augmentor, we define the augmentation in here
        :param image: image of dataset
        :param label: label of dataset
        :param weight: sample weight
        :return: the augmented image and label, and the sample weight
        """

        image = image.numpy()
        label = label.numpy()
        weight = weight.numpy()

        transforms = Compose([
            A.Rotate(limit=self.rotation_range, p=self.rotation_proba),
            A.HorizontalFlip(p=self.flip),
            A.Transpose(p=0.5),
            # A.OneOf([
            #     A.MotionBlur(p=0.2),
            #     A.MedianBlur(blur_limit=3, p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            # ], p=0.2),
            # A.OpticalDistortion(p=0.3),
            # # A.CLAHE(clip_limit=2, p=1),
            # A.OneOf([
            #     A.IAASharpen(p=0.5),
            #     A.IAAEmboss(p=0.5),
            #     A.RandomBrightnessContrast(p=0.5),
            # ], p=0.3)
        ], additional_targets={
            'image1': 'image',
            'mask1': 'mask',
            'mask2': 'mask'
        })

        aug_data = {"image": image,
                    "mask1": label,
                    "mask2": weight}
        aug_data = transforms(**aug_data)
        aug_img, aug_label, aug_weight = aug_data["image"], aug_data["mask1"], aug_data["mask2"]
        # print('np.amax(aug_label):', np.amax(aug_label))

        return aug_img, aug_label, aug_weight

    def process_data(self, image, label, weight):
        """
        this method is for calling a aug_fn, which is a function that takes numpy array as input
        :param image: image of the dataset
        :param label:  label of the dataset
        :param weight: smaple weight
        :return: a tuple that consists of augmented image,label,sample weight
        """
        aug_img = tf.py_function(self.aug_fn, (image, label, weight), (tf.float32, tf.float32, tf.float64))
        return aug_img

    def add_augmentation(self, data):

        """
        calling the augmentation and map the dataset to the augmentation methods
        :param data: train dataset, type=tf.dataset
        :return: augmented train dataset
        """
        if self.do_aug_train:
            data = data.map(self.process_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        return data

    def add_validation_augmentation(self, data):

        """
        calling the augmentation and map the dataset to the augmentation methods
        :param data: val dataset, type=tf.dataset
        :return: augmented val dataset
        """
        if self.do_aug_val:
            data.map(partial(self.process_data), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        return data


import os
import sys
import yaml

# class Struct:
#     def __init__(self, **entries):
#         for k, v in entries.items():
#             if isinstance(v, dict):
#                 self.__dict__[k] = Struct(**v)
#             else:
#                 self.__dict__[k] = v

# utils_dir = os.path.abspath('../../runs/template/config.yaml')
# sys.path.append(utils_dir)
# print(utils_dir)
#
# with open(utils_dir) as f:
#     # use safe_load instead load
#     data_map = yaml.safe_load(f)
#
# config = Struct(**data_map)
#
# dataset = EchoNetDataLoader(config)
# train_gen, n_iter_train = dataset.create_train_data_generator()
# print(type(train_gen))
#
# augmenatation_ = Aug()
# train_gen = augmenatation_.add_augmentation(train_gen)
# print(type(train_gen))
# batch = next(iter(train_gen))
# for i, ele in zip(range(0, 1), train_gen):
#     print('index:', i)
#     print(ele)
#     print('len(ele):', len(ele))
#     print('ele[0].numpy().shape:', ele[0].numpy().shape)
#     first_img = ele[0]
#     img_label = ele[1]
#     img_weights = ele[2]
#     fig, ax = plt.subplots(1, 3)
#     ax[0].imshow(first_img)
#     ax[1].imshow(img_label)
#     ax[2].imshow(img_weights)
#     plt.show()
