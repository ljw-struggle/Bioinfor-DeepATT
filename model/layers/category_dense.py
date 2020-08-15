# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class CategoryDense(keras.layers.Layer):
    """ CategoryDense
    """
    def __init__(self,
                 units,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Initialize the Category Dense layer.
        :param units: num of hidden units.
        """
        super(CategoryDense, self).__init__(**kwargs)
        self.units = units
        self.kernel = None
        self.bias = None
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        super(CategoryDense, self).build(input_shape=input_shape)
        category = input_shape[1]
        input_channel = input_shape[2]
        output_channel = self.units
        kernel_shape = [1, category, input_channel, output_channel]
        bias_shape = [1, category, output_channel]
        self.kernel = self.add_weight(
            shape=kernel_shape,
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.bias = self.add_weight(
            shape=bias_shape,
            name='bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        self.built = True


    def call(self, inputs, **kwargs):
        """
        Call function of Category Dense layer.
        :param inputs: shape = (batch_size, Categories, channel)
        :return: shape = (batch_size, Categories, output_channel)
        """
        inputs = inputs[:, :, :, tf.newaxis]
        outputs = tf.reduce_sum(tf.multiply(inputs, self.kernel), axis=2)
        outputs = tf.add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)

        return outputs