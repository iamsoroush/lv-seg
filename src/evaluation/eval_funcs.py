import tensorflow as tf


def get_conf_mat_elements(y_true, y_pred, threshold=0.5):

    """Returns true positives count.

    Args:
        y_true (tf.Tensor): Tensor of shape(h, w, 1) with binary elements.
        y_pred (tf.Tensor): Tensor of shape(h, w, 1), and range(0, 1) which is prediction of a segmentation model.

    Returns:
        tuple(tp, tn, fp, fn):
            - tp: true positives
            - tn: true negatives
            - fp: false positives
            - fn: false negatives

    """

    y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)
    y_true_thresholded = tf.cast(y_true > threshold, tf.float32)

    conf_mat = tf.math.confusion_matrix(tf.reshape(y_true_thresholded, -1), tf.reshape(y_pred_thresholded, -1))
    tn, fp, fn, tp = tf.reshape(conf_mat, -1)
    return tp, tn, fp, fn


def get_tpr(threshold=0.5):

    def true_positive_rate(y_true, y_pred):

        """Calculates true positive rate

        Args:
            y_true (tf.Tensor): Tensor of shape(h, w, 1) with binary elements.
            y_pred (tf.Tensor): Tensor of shape(h, w, 1), and range(0, 1) which is prediction of a segmentation model.

        Returns:
            the percentage of the true positives
        """

        tp, tn, fp, fn = get_conf_mat_elements(y_true, y_pred, threshold)
        return float((tp / (tp + fn)) * 100)

    f = true_positive_rate
    f.__name__ += f'_th{threshold}'

    return f


def get_tnr(threshold=0.5):

    def true_negative_rate(y_true, y_pred):

        """Calculates true negative rate

        Args:
            y_true (tf.Tensor): Tensor of shape(h, w, 1) with binary elements.
            y_pred (tf.Tensor): Tensor of shape(h, w, 1), and range(0, 1) which is prediction of a segmentation model.

        Returns:
            the percentage of the true negative
        """

        tp, tn, fp, fn = get_conf_mat_elements(y_true, y_pred, threshold)
        return float((tn / (fp + tn)) * 100)

    f = true_negative_rate
    f.__name__ += f'_th{threshold}'

    return f


def get_iou_coef(threshold=0.5, smooth=0.001):

    def iou_coef(y_true, y_pred):
        """Intersection over Union for y_true and y_pred

        Args:
            y_true (tf.Tensor): Tensor of shape(h, w, 1) with binary elements.
            y_pred (tf.Tensor): Tensor of shape(h, w, 1), and range(0, 1) which is prediction of a segmentation model.

        Returns:
            Intersection over Union for y_true and y_pred
        """

        #  keras uses float32 instead of float64,
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        # our input y_pred is softmax prediction so we change it to 0 ,1 classes
        y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)

        intersection = tf.math.reduce_sum(tf.abs(y_true * y_pred_thresholded))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_thresholded) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

    f = iou_coef
    f.__name__ += f'_th{threshold}'

    return f


def get_soft_iou(smooth=0.001):

    def soft_iou(y_true, y_pred):

        """
        Args:
            y_true (tf.Tensor): Tensor of shape(h, w, 1) with binary elements.
            y_pred (tf.Tensor): Tensor of shape(h, w, 1), and range(0, 1) which is prediction of a segmentation model.

        Returns:
            soft intersection over Union for y_true and y_pred
        """

        #  keras uses float32 instead of float64,
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

    f = soft_iou

    return f


def get_dice_coeff(threshold=0.5, smooth=0.001):

    def dice_coef(y_true, y_pred):
        """Calculate dice coefficient between y_true and y_pred

        Args:
            y_true (tf.Tensor): Tensor of shape(h, w, 1) with binary elements.
            y_pred (tf.Tensor): Tensor of shape(h, w, 1), and range(0, 1) which is prediction of a segmentation model.

        Returns:
            dice coefficient between y_true and y_pred
        """

        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred_thresholded)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_thresholded)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    f = dice_coef
    f.__name__ += f'_th{threshold}'

    return f


def get_soft_dice(epsilon=1e-6):
    """
    Args:
        epsilon: Used for numerical stability to avoid divide by zero errors
    """

    def soft_dice(y_true, y_pred):
        """Soft dice loss calculation. Assumes the channels_last format.

        Args:
            y_true (tf.Tensor): Tensor of shape(h, w, 1) with binary elements.
            y_pred (tf.Tensor): Tensor of shape(h, w, 1), and range(0, 1) which is prediction of a segmentation model.

        Returns:
            dice coefficient between y_true and y_pred
        """

        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        numerator = 2. * tf.reduce_sum(y_pred * y_true)
        denominator = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true))

        return (numerator + epsilon) / (denominator + epsilon)

    f = soft_dice

    return f
