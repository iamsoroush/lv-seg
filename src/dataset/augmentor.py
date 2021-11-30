from abstractions import AugmentorBase
import albumentations as A
import numpy as np


class Augmentor(AugmentorBase):

    """
    This class is implementing augmentation on the batches of data

    Example::

        aug = Augmentor(config)
        augmented_gen = aug.add_augmentation(generator)
        augmented_val_gen = aug.add_validation_augmentation(generator)

    Augmentation part of config file:

        config.pre_process.augmentation, containing:

            rotation_range - the range limitation for rotation in augmentation
            flip_proba - probability for flipping

    """

    def __init__(self, config=None):
        super().__init__(config)

        self.transform = A.Compose([
            A.Flip(p=self.flip_proba),
            A.ShiftScaleRotate(0, 0, border_mode=0, rotate_limit=self.rotation_range, p=self.rotation_proba)
        ])

    def batch_augmentation(self, batch):

        """This method implement augmentation on batches

        :param batch: (x, y, z):
            x: batch images of the whole batch
            y: batch masks of the whole batch
            z: batch third element (index for test and sample weight for training set and val set
        :return x: image batch
        :return y: mask batch.
        :return z: third element of batch.
        """

        # changing the type of the images for albumentation
        x = batch[0]
        y = batch[1]
        z = batch[2]

        if 'list' in str(type(x)) or 'list' in str(type(y)):
            x = np.array(x, dtype='float32')
            y = np.array(y, dtype='float32')
        else:
            x = x.astype('float32')

        # implementing augmentation on every image and mask of the batch
        for i in range(len(x)):
            transformed = self.transform(image=x, mask=y)
            x = transformed['image']
            y = transformed['mask']

        return x, y, z

    def add_augmentation(self, generator):
        """

        Args:
            generator: a generator with batch size of 1 sample that consist of a three element tuple
            (image, label, third_element)

        Returns:
            a generator in the shape of input with augmented image and label
        """
        while True:
            batch = next(generator)
            augmented_batch = self.batch_augmentation(batch)
            yield augmented_batch

    def add_validation_augmentation(self, generator):
        """
        This method is used whenever we have a unique approach for augmenting validation set
        Args:
            generator: a generator with batch size of 1 sample that consist of a three element tuple
            (image, label, third_element)

        Returns:
            a generator in the shape of input with augmented image and label
        """
        pass

    def _load_params(self, config):

        aug_config = config.augmentator
        self.rotation_range = aug_config.rotation_range
        self.rotation_proba = aug_config.rotation_proba
        self.flip_proba = aug_config.flip_proba

    def _set_defaults(self):

        self.rotation_range = 45
        self.rotation_proba = 0.5
        self.flip_proba = 0.5
