# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import scipy.io as sio

class SequenceLoader(object):
    def __init__(self, data_dir='./data/', batch_size=64, shuffle=True, num_workers=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_steps = int(np.ceil(4400000/batch_size))
        self.valid_steps = int(np.ceil(8000/100))
        self.test_steps = int(np.ceil(455024/100))

    def get_train_data(self):
        filenames = ['./data/traindata-00.tfrecord', './data/traindata-01.tfrecord',
                     './data/traindata-02.tfrecord', './data/traindata-03.tfrecord']
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=None)
        if self.shuffle == True:
            dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        dataset = dataset.map(map_func=self.parse_function, num_parallel_calls=self.num_workers)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=10000)
        return dataset # 4400000/64 = 68750

    def get_valid_data(self):
        data = sio.loadmat('./data/valid.mat')
        x = data['validxdata']  # shape = (8000, 4, 1000)
        y = data['validdata']  # shape = (8000, 919)
        x = np.transpose(x, (0, 2, 1)).astype(dtype=np.float32)  # shape = (8000, 1000, 4)
        y = np.transpose(y, (0, 1)).astype(dtype=np.int32)  # shape = (8000, 919)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(100)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset # 8000/100 = 80

    def get_test_data(self):
        filenames = ['./data/testdata.tfrecord']
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000, num_parallel_reads=None)
        dataset = dataset.map(map_func=self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(100, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset # 455024/64 = 7109.75 = 7110

    @staticmethod
    def parse_function(example_proto):
        dics = {
            'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
            'y': tf.io.FixedLenFeature([919], tf.int64),
        }
        parsed_example = tf.io.parse_single_example(example_proto, dics)
        x = tf.reshape(parsed_example['x'], [1000, 4])
        y = tf.reshape(parsed_example['y'], [919])
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int32)
        return (x, y)