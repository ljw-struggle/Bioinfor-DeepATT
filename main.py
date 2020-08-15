# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf

from utils.configer import ConfigParser

def train(configer):
    configer.set_reproducibility()
    loader = configer.get_loader()
    trainer = configer.get_trainer()
    trainer.train(
        loader.get_train_data(),
        loader.get_valid_data(),
        train_steps=loader.train_steps,
        valid_steps=loader.valid_steps)

def test(configer):
    configer.set_reproducibility()
    loader = configer.get_loader()
    trainer = configer.get_trainer()
    trainer.test(
        loader.get_test_data(),
        test_steps=loader.test_steps)

if __name__ == '__main__':
    # 1\ Parses the command line arguments and returns as a simple namespace.
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.(train/test)')
    parser.add_argument('-c', '--config', default='./config/config_0.json', help='The config file of experiment.')
    parser.add_argument('-i', '--identifier', default=0, help='The run id of experiment.')
    parser.add_argument('-v', '--verbosity', default=0, help='The verbosity of training/testing process.')
    args = parser.parse_args()

    # 2\ Configure the Check the Environment.
    tf.debugging.set_log_device_placement(False)
    tf.config.set_soft_device_placement(True)
    cpu_devices = tf.config.experimental.list_physical_devices('CPU')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    print('Check the Deep learning Environment:', flush=True)
    print('GPU count:{}, Memory growth:{}, Soft device placement:{} ...'.format(len(gpu_devices),True,True), flush=True)

    # 3\ Get the configer.
    configer = ConfigParser.from_config_file(args.config, args.identifier, args.verbosity)

    # 4\ Selecting the execution mode.
    if args.exe_mode == 'train':
        train(configer = configer)
    elif args.exe_mode == 'test':
        test(configer = configer)
    else:
        print('No mode named {}.'.format(args.exe_mode))