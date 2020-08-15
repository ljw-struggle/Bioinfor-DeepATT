# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class BidLSTM(keras.layers.Layer):
    """ Bidirectional LSTM Layer.

        Reference:
            - [LSTM](https://arxiv.org/abs/1402.1128)
    """
    def __init__(self, units=100):
        """
        Initialize the BidLSTM layer.
        :param units: num of hidden units.
        """
        super(BidLSTM, self).__init__()
        forward_layer = keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True)
        backward_layer = keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True,
            go_backwards=True)
        self.bidirectional_rnn = keras.layers.Bidirectional(
            layer=forward_layer,
            backward_layer=backward_layer)

    def build(self, input_shape):
        super(BidLSTM, self).build(input_shape=input_shape)

    def call(self, inputs, mask = None, **kwargs):
        """
        Call function of BidLSTM layer.
        :param inputs: shape = (batch_size, time_steps, channel)
        :param mask: shape = (batch_size, time_steps)
        :param kwargs: None.
        :return: (sequence_output, state_output).
                  sequence_output shape is (batch_size, time_steps, units x 2),
                  state_output shape is (batch_size, units x 2)
        """
        output = self.bidirectional_rnn(inputs, mask=mask)
        sequence_output = output[0]
        forward_state_output = output[1]
        backward_state_output = output[2]
        state_output = tf.keras.layers.concatenate([forward_state_output, backward_state_output], axis=-1)
        return sequence_output, state_output

    @staticmethod
    def create_padding_mask(seq_len, max_len):
        """
        Create the padding mask matrix according to the seq_len and max_len.
        Set the value to 0 to mask the padding sequence.
        :param seq_len: the sequence length.
        :param max_len: the max length.
        :return: padding mask matrix. (shape = (batch_size, max_len))
        """
        mask_matrix = tf.sequence_mask(seq_len, maxlen=max_len)
        return mask_matrix


class BidGRU(keras.layers.Layer):
    """ Bidirectional GRU Layer.

        Reference:
            - [Gated Recurrent Unit](https://arxiv.org/abs/1406.1078)
    """
    def __init__(self, units=100):
        """
        Initialize the BidGRU layer.
        :param units: num of hidden units.
        """
        super(BidGRU, self).__init__()
        forward_layer = keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True)
        backward_layer = keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            go_backwards=True)
        self.bidirectional_rnn = keras.layers.Bidirectional(
            layer=forward_layer,
            backward_layer=backward_layer)

    def build(self, input_shape):
        super(BidGRU, self).build(input_shape=input_shape)

    def call(self, inputs, mask = None, **kwargs):
        """
        Call function of BidGRU layer.
        :param inputs: shape = (batch_size, time_steps, channel)
        :param mask: shape = (batch_size, time_steps)
        :param kwargs: None.
        :return: (sequence_output, state_output).
                  sequence_output shape is (batch_size, time_steps, units x 2),
                  state_output shape is (batch_size, units x 2)
        """
        output = self.bidirectional_rnn(inputs, mask=mask)
        sequence_output = output[0]
        forward_state_output = output[1]
        backward_state_output = output[2]
        state_output = tf.keras.layers.concatenate([forward_state_output, backward_state_output], axis=-1)
        return sequence_output, state_output

    @staticmethod
    def create_padding_mask(seq_len, max_len):
        """
        Create the padding mask matrix according to the seq_len and max_len.
        Set the value to 0 to mask the padding sequence.
        :param seq_len: the sequence length.
        :param max_len: the max length.
        :return: padding mask matrix. (shape = (batch_size, max_len))
        """
        mask_matrix = tf.sequence_mask(seq_len, maxlen=max_len)
        return mask_matrix