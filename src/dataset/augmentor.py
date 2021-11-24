from abstractions import AugmentorBase
import albumentations as A
import numpy as np


class Augmentor(AugmentorBase):

    """
    This class is implementing augmentation on the batches of data

    Example::

        aug = Augmentation()
        data = aug.batch_augmentation(x,y)
        x = images(batch)
        y = masks(batch)
        data = augmented batch

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

        :param batch: (x, y):
            x: batch images of the whole batch
            y: batch masks of the whole batch

        :return x: image batch
        :return y: mask batch.
        """

        # changing the type of the images for albumentation
        x = batch[0]
        y = batch[1]

        if 'list' in str(type(x)) or 'list' in str(type(y)):
            x = np.array(x, dtype='float32')
            y = np.array(y, dtype='float32')
        else:
            x = x.astype('float32')

        # implementing augmentation on every image and mask of the batch
        for i in range(len(x)):
            transformed = self.transform(image=x[i], mask=y[i])
            x[i] = transformed['image']
            y[i] = transformed['mask']

        return x, y

    def add_augmentation(self, generator):

        """

        Args:
            generator:

        Returns:

        """
        while True:
            batch = next(generator)
            augmented_batch = self.batch_augmentation(batch)
            yield augmented_batch

    def add_validation_augmentation(self, generator):
        pass

    def _load_params(self, config):

        aug_config = config.augmentator

        self.rotation_range = aug_config.rotation_range
        self.rotation_proba = aug_config.rotation_proba
        self.flip_proba = aug_config.flip_proba
        self.do_train_augmentation = config.do_train_augmentation
        self.do_validation_augmentation = config.do_validation_augmentation

    def _set_defaults(self):

        self.rotation_range = 45
        self.rotation_proba = 0.5
        self.flip_proba = 0.5
        self.do_train_augmentation = True
        self.do_validation_augmentation = False
