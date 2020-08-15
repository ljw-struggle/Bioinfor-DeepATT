# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class GroupNormalization(keras.layers.Layer):
    """ Group normalization layer

        Group Normalization divides the channels into groups and computes within each group
        the mean and variance for normalization. GN's computation is independent of batch sizes,
        and its accuracy is stable in a wide range of batch sizes.

        Mechanism
            1\ Group Normalization.
            2\ Inverse Normalization.

        References
            - [Group Normalization](https://arxiv.org/abs/1803.08494)
            - https://github.com/shaoanlu/GroupNormalization-keras/blob/master/GroupNormalization.py
            - https://github.com/titu1994/Keras-Group-Normalization/blob/master/group_norm.py
            - https://github.com/jiawei6636/3d-brain-tumor-segmentation/blob/develop/layers/group_norm.py
    """
    def __init__(self,
                 groups=8,
                 axis=-1,
                 epsilon=1e-5,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        """
        Initial function of Group Normalization.
        :param groups: Integer, the number of groups for Group Normalization.
        :param axis: Integer, the axis that should be normalized.(Only support Channel_last, axis = -1)
        :param epsilon: Small float added to variance to avoid dividing by zero.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param kwargs: None
        """
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)


    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

        shape = [1] * len(input_shape)
        shape[self.axis] = dim
        self.gamma = self.add_weight(
            shape=shape,
            name='gamma',
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint)
        self.beta = self.add_weight(
            shape=shape,
            name='beta',
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint)
        self.built = True


    def call(self, inputs, **kwargs):
        """
        Call function of Group Normalization. (Only support 'Channel last'.)
        :param inputs: shape = (batch_size, ..., channel)
        :param kwargs: None
        :return: (batch_size, ..., channel)
        """
        input_shape = list(inputs.shape) # input_shape = (batch_size, ..., channel)
        input_shape[0] = -1

        # 1\ Group Normalization.
        group_shape = list(inputs.shape) # group_shape = (batch_size, ..., channel)
        group_shape[0] = -1
        group_shape[self.axis] = group_shape[self.axis]//self.groups
        group_shape.insert(self.axis, self.groups) # group_shape = (batch_size, ..., groups, c//groups)

        group_axes = list(range(len(group_shape))) # group_axes = [0, ..., -2, -1]
        group_axes = group_axes[1:-2] + [group_axes[self.axis]] # group_axes = [..., -1]

        inputs = tf.reshape(inputs, tuple(group_shape))
        mean, variance = tf.nn.moments(inputs, axes=group_axes, keepdims=True)

        inputs = (inputs - mean)/tf.math.sqrt(variance + self.epsilon)
        normal_outputs = tf.reshape(inputs, input_shape) # shape = (batch_size, ..., channel)

        # 2\ Inverse Group Normalization.
        inverse_outputs = normal_outputs * self.gamma + self.beta # shape = (batch_size, ..., channel)

        return inverse_outputs


    def get_config(self):
        config = super(GroupNormalization, self).get_config()
        config.update({'groups': self.groups,
                       'axis': self.axis,
                       'epsilon': self.epsilon,
                       'beta_initializer': keras.initializers.serialize(self.beta_initializer),
                       'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
                       'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
                       'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
                       'beta_constraint': keras.constraints.serialize(self.beta_constraint),
                       'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)})
        return config


    def compute_output_shape(self, input_shape):
        return input_shape