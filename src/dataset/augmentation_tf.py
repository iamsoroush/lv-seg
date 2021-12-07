from functools import partial
import tensorflow as tf
from abstractions.augmentation import AugmentorBase
from tensorflow.python.data import AUTOTUNE
import albumentations as A
from albumentations import (
    Compose)


class Augmentor(AugmentorBase):
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
        Returns:
             config: config file
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
        Args:
            image: image of dataset
            label: label of dataset
            weight: sample weight
        Returns:
            the augmented image and label, and the sample weight
        """

        image = image.numpy()
        label = label.numpy()
        weight = weight.numpy()

        transforms = Compose([
            A.Rotate(limit=self.rotation_range, p=self.rotation_proba),
            A.HorizontalFlip(p=self.flip),
            A.Transpose(p=0.5),
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
        Args:
            image: image of the dataset
            label:  label of the dataset
            weight: smaple weight
        Returns:
            a tuple that consists of augmented image,label,sample weight
        """
        aug_img = tf.py_function(self.aug_fn, (image, label, weight), (tf.float32, tf.float32, tf.float64))
        return aug_img

    def add_augmentation(self, data):

        """
        calling the augmentation and map the dataset to the augmentation methods
        Args:
            data: train dataset, type=tf.dataset
        Returns:
            augmented train dataset
        """
        if self.do_aug_train:
            data = data.map(self.process_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        return data

    def add_validation_augmentation(self, data):

        """
        calling the augmentation and map the dataset to the augmentation methods
        Args:
            data: val dataset, type=tf.dataset
        Returns:
            augmented val dataset
        """
        if self.do_aug_val:
            data.map(partial(self.process_data), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        return data
