# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class BinaryFocalloss(keras.losses.Loss):
    """
    Binary Focal Loss. (Implemented by Jiawei Li)
    Implementation of focal loss. (<https://arxiv.org/pdf/1708.02002.pdf>)(Kaiming He)
    Binary Focal Loss Formula: FL = - y_true * alpha * (1-y_pred)^gamma * log(y_pred)
                             - (1 - y_true) * (1-alpha) * y_pred^gamma * log(1-y_pred)
                        ,which alpha = 0.25, gamma = 2, y_pred = sigmoid(x), y_true = target_tensor,
                        y_pred.shape = (batch_size, 1), y_true.shape = (batch_size, 1).
    """
    def __init__(self,
                 smoothing=0,
                 alpha=0.25,
                 gamma=2,
                 name='binary_focalloss',
                 **kwargs):
        """
        Initializes Binary Focal Loss class and sets attributes needed in loss calculation.
        :param smoothing: float, optional amount of label smoothing to apply. Set to 0 for no smoothing.
        :param alpha: float, optional amount of balance to apply (as in balanced cross entropy).
        :param gamma: int, optional amount of focal smoothing to apply.
                      Set to 0 for regular balanced cross entropy.
        :param name: str, optional name of this loss class (for tf.Keras.losses.Loss).
        :param kwargs: {'reduction': tf.keras.losses.Reduction.AUTO}
        """
        super(BinaryFocalloss, self).__init__(name = name, **kwargs)
        assert smoothing <= 1 and smoothing >= 0, '`smoothing` needs to be in the range [0, 1].'
        assert alpha <= 1 and alpha >= 0, '`alpha` needs to be in the range [0, 1].'
        assert gamma >= 0, '`gamma` needs to be a non-negative integer.'
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Computes binary focal loss between predicted probabilities and true label.
        :param y_true: ground truth labels. shape = (batch_size, 1)
        :param y_pred: predicted probabilities (softmax or sigmoid). shape = (batch_size, 1)
        :return: focal loss.
        """
        y_true = tf.cast(y_true, tf.float32)

        # 1\ Label Smoothing.
        if self.smoothing > 0:
            y_true = y_true * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # 2\ Clip values for Numerical Stable. (Avoid NaN calculations)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # 3\ Calculate the focal loss.
        focal = y_true * self.alpha * tf.pow((1-y_pred), self.gamma) * tf.math.log(y_pred)
        focal += (1 - y_true) * (1-self.alpha) * tf.pow(y_pred, self.gamma) * tf.math.log(1 - y_pred)
        loss = -focal

        # 4\ Sample Weight and Reduction
        # Note: sample_weight and reduction are implemented in the __call__ function.
        # In the super class tf.keras.losses.Loss, the __call__ function will invoke the call function.

        return loss


class CategoricalFocalloss(keras.losses.Loss):
    """
    Categorical Focal Loss, for multi-class classification. (Implemented by Jiawei Li)
    Implementation of focal loss. (<https://arxiv.org/pdf/1708.02002.pdf>)(Kaiming He)
    1\ Method 1: (Official) (we use this method)
        Categorical Focal Loss Formula: FL = - sum(y_true * alpha * ((1-y_pred)^gamma)*log(y_pred), -1)
                        ,which alpha.shape = (classes), y_pred.shape = (batch_size, classes),
                        y_true.shape = (batch_size, classes), gamma = 2, y_pred = softmax(logits).
    2\ Method 2: (achieve it by multiple binary focal loss, more suitable for multi-label classification)
        Categorical Focal Loss Formula: FL = sum(- y_true * alpha * (1-y_pred)^gamma * log(y_pred)
                                                    - (1 - y_true) * (1-alpha) * y_pred^gamma * log(1-y_pred), -1)
                        ,which alpha.shape = (classes), y_pred.shape = (batch_size, classes),
                        y_true.shape = (batch_size, classes), gamma = 2, y_pred = sigmoid(logits).
    """
    def __init__(self,
                 smoothing=0,
                 alpha=None,
                 gamma=2,
                 name='categorical_focalloss',
                 **kwargs):
        """
        Initializes Categorical Focal Loss class and sets attributes needed in loss calculation.
        :param smoothing: float, optional amount of label smoothing to apply. Set to 0 for no smoothing.
        :param alpha: list, (sum to 1). optional amount of balance to apply (as in balanced cross entropy).
        :param gamma: int, optional amount of focal smoothing to apply. Set to 0 for regular balanced cross entropy.
        :param name: str, optional name of this loss class (for tf.Keras.losses.Loss).
        :param kwargs:
        """
        super(CategoricalFocalloss, self).__init__(name = name, **kwargs)
        assert smoothing <= 1 and smoothing >= 0, '`smoothing` needs to be in the range [0, 1].'
        assert gamma >= 0, '`gamma` needs to be a non-negative integer.'
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Computes categorical focal loss between predicted probabilities and true label.
        :param y_true: ground truth labels. shape = (batch_size, ..., classes)
        :param y_pred: predicted probabilities (softmax or sigmoid). shape = (batch_size, ..., classes)
        :return: focal loss.
        """
        y_true = tf.cast(y_true, tf.float32)
        k = tf.shape(y_true)[-1]

        # 1\ Label Smoothing.
        if self.smoothing > 0:
            y_true = y_true * (1.0 - self.smoothing) + 1/k * self.smoothing

        # 2\ Clip values for Numerical Stable. (Avoid NaN calculations)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # 3\ Check the dimensionality of alpha and k.
        if self.alpha == None:
            alpha = tf.cast(1/k, tf.float32)
        else:
            alpha = tf.cast(self.alpha, tf.float32)
            assert tf.equal(tf.shape(self.alpha)[0], k), 'the dimensionality of alpha is not correct!'

        # 4\ Calculate Categorical Focal Loss.
        focal = y_true * alpha * tf.pow((1-y_pred), self.gamma) * tf.math.log(y_pred)
        loss = -tf.reduce_sum(focal, -1)

        # 5\ Sample Weight and Reduction
        # Note: sample_weight and reduction are implemented in the __call__ function.
        # In the super class tf.keras.losses.Loss, the __call__ function will invoke the call function.

        return loss