from functools import partial

import tensorflow as tf

from abc import ABC
from abstractions.augmentor import AugmentorBase
from tensorflow.python.data import AUTOTUNE

from data_loader import EchoNetDataLoader
import albumentations as A
from albumentations import (
    Compose)


class Aug(AugmentorBase, ABC):
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

    def aug_fn(self, image, label):
        """
        this method is the augmentor, we define the augmentation in here
        :param image: image of dataset
        :param label: label of dataset
        :param weight: sample weight
        :return: the augmented image and label, and the sample weight
        """
        print("badddddddd")
        transforms = Compose([
            A.Rotate(limit=self.rotation_range, p=self.rotation_proba),
            A.HorizontalFlip(p=self.flip),
            A.Transpose(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OpticalDistortion(p=0.3),
            # A.CLAHE(clip_limit=2, p=1),
            A.OneOf([
                A.IAASharpen(p=0.5),
                A.IAAEmboss(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], p=0.3)
        ])
        print("goooooddddddd111111")
        image = {"image": image}
        label = {"image": label}
        aug_data = transforms(image, label)
        print(type(aug_data))
        print("hiiiiiiiiiiiiiiiiiiiiiiii")
        aug_img = aug_data["image","mask"]
        aug_img = tf.cast(aug_img, tf.float32)

        return aug_img

    def process_data(self, image, label):
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[image, label], Tout=tf.float32)
        print("gooooodddddd")
        return aug_img

    @staticmethod
    def image_augmentation(image, label):
        """

        :param image:
        :param label:
        :return:
        """
        image = tf.image.flip_left_right(image)
        image = tf.image.random_contrast(image, 0.2, 0.5)

        return image, label

    def add_augmentation(self, data):

        """
        calling the batch_augmentation
        :param data:
        :param tftensor: the input of this class must be generator
        :yield: the batches of the augmented generator
        """
        if self.do_aug_train:
            print("goodddddd")
        data=data.map(partial(self.process_data), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        print("goodddddd")
        return data

    def add_validation_augmentation(self, data):

        """
        calling the batch_augmentation
        :param tftensor: the input of this class must be generator
        :yield: the batches of the augmented generator
        """
        if self.do_aug_val:
            data.map(partial(self.process_data), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        return data

import os
import sys
import yaml

class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v

utils_dir = os.path.abspath('../../runs/template/config.yaml')
sys.path.append(utils_dir)
print(utils_dir)

with open(utils_dir) as f:
    # use safe_load instead load
    data_map = yaml.safe_load(f)

config = Struct(**data_map)

dataset = EchoNetDataLoader(config)
train_gen, n_iter_train= dataset.create_train_data_generator()
print(type(train_gen))

augmenatation = Aug(None)
train_gen = augmenatation.add_augmentation(train_gen)
print(type(train_gen))