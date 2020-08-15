# -*- coding: utf-8 -*-
import h5py
import numpy as np
import tensorflow as tf
import scipy.io as sio

from tqdm import tqdm

def serialize_example(x, y):
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    example = {
        'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.flatten())),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))}

    # Create a Features message using tf.train.Example.
    example = tf.train.Features(feature=example)
    example = tf.train.Example(features=example)
    serialized_example = example.SerializeToString()
    return serialized_example

def traindata_to_tfrecord():
    filename = './data/train.mat'
    with h5py.File(filename, 'r') as file:
        x = file['trainxdata'] # shape = (1000, 4, 4400000)
        y = file['traindata'] # shape = (919, 4400000)
        x = np.transpose(x, (2, 0, 1)) # shape = (4400000, 1000, 4)
        y = np.transpose(y, (1, 0)) # shape = (4400000, 919)

    for file_num in range(4):
        with tf.io.TFRecordWriter('./data/traindata-%.2d.tfrecord' % file_num) as writer:
            for i in tqdm(range(file_num*1100000, (file_num+1)*1100000), desc="Processing Train Data {}".format(file_num), ascii=True):
                example_proto = serialize_example(x[i], y[i])
                writer.write(example_proto)

def testdata_to_tfrecord():
    filename = './data/test.mat'
    data = sio.loadmat(filename)
    x = data['testxdata'] # shape = (455024, 4, 1000)
    y = data['testdata'] # shape = (455024, 919)
    x = np.transpose(x, (0, 2, 1)) # shape = (455024, 1000, 4)
    y = np.transpose(y, (0, 1)) # shape = (455024, 919)

    with tf.io.TFRecordWriter('./data/testdata.tfrecord') as writer:
        for i in tqdm(range(len(y)), desc="Processing Test Data", ascii=True):
            example_proto = serialize_example(x[i], y[i])
            writer.write(example_proto)

if __name__ == '__main__':
    # Write the train data and test data to .tfrecord file.
    traindata_to_tfrecord()
    testdata_to_tfrecord()