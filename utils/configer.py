# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

from datetime import datetime
from utils.utils import read_json, write_json, create_dirs

class ConfigParser:
    def __init__(self, config, run_id=None, verbosity=0):
        """
        Class to parse configuration json file. Handles hyper-parameters for training, initializations of modules,
        checkpoint saving and logging module.
        :param config: Dict containing configurations.
        :param run_id: Unique Identifier for training processes. Timestamp is being used as default
        :param verbosity: default 0.
        """
        # 1\ Define the config and run_id.
        self._config = config
        self._run_id = str(run_id)
        if run_id is 'None':
            self._run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._verbosity = verbosity

        # 2\ Set _save_dir, _checkpoint_dir and _log_dir where checkpoints and logs will be saved.
        save_dir = './result/'
        self._result_dir = os.path.join(save_dir, self.config['name']+'/', self._run_id+'/')
        self._checkpoint_dir = os.path.join(save_dir, self.config['name']+'/', self._run_id+'/', 'checkpoints/')
        self._summary_dir = os.path.join(save_dir, self.config['name']+'/', self._run_id+'/', 'logs/')

        # 3\ Create directory for saving checkpoints and log.
        create_dirs([self.result_dir, self._checkpoint_dir, self.summary_dir])

        # 4\ Save relative config file to the relative dir
        write_json(self.config, os.path.join(self.result_dir, 'config.json'))
        self.config['trainer']['args']['verbosity'] = verbosity

    def __getitem__(self, name):
        # Access items like ordinary dict.
        return self.config[name]

    def set_reproducibility(self):
        np.random.seed(self['seed'])
        tf.random.set_seed(self['seed'])

    def get_loader(self):
        from utils import loader_dict
        return loader_dict[self['loader']['type']](self['loader']['args'])

    def get_trainer(self):
        from model import model_dict, loss_dict
        from optimizer import optimizer_dict, scheduler_dict
        from utils import trainer_dict

        message = "The learning rate of optimizer and scheduler should be same."
        learning_rate = self['optimizer']['args']['learning_rate']
        if self['lr_scheduler']['type'] != None:
            assert learning_rate==self['lr_scheduler']['args']['initial_learning_rate'], message

        hparams = {
            'model': self['model']['type'],
            'loss': self['loss']['type'],
            'optimizer': self['optimizer']['type'],
            'batch_size': self['loader']['args']['batch_size'],
            'learning_rate': learning_rate,
            'lr_scheduler': str(self['lr_scheduler']['type'])
        }

        model = model_dict[self['model']['type']](**self['model']['args'])
        model.build(input_shape=(None, 1000, 4))
        model.summary()
        loss_object = loss_dict[self['loss']['type']](**self['loss']['args'])
        if self['lr_scheduler']['type'] == None:
            optimizer = optimizer_dict[self['optimizer']['type']](**self['optimizer']['args'])
        else:
            lr_scheduler = scheduler_dict[self['lr_scheduler']['type']](**self['lr_scheduler']['args'])
            self['optimizer']['args']['learning_rate'] = lr_scheduler
            optimizer = optimizer_dict[self['optimizer']['type']](**self['optimizer']['args'])

        trainer = trainer_dict[self['trainer']['type']](
            model=model,
            loss_object=loss_object,
            optimizer=optimizer,
            hparams = hparams,
            result_dir=self.result_dir,
            checkpoint_dir = self.checkpoint_dir,
            summary_dir = self.summary_dir,
            **self['trainer']['args'])
        return trainer

    @classmethod
    def from_config_file(cls, config_file, identifier, verbosity):
        """
        Initialize this class from config file. Used in train, test.
        :param config_file: config file.
        :param identifier: identifier
        :return: ConfigParser.
        """
        message = "Configuration file need to be specified. Add '-c ./config/config.json', for example."
        assert config_file is not None, message
        config = read_json(config_file)
        run_id = identifier

        # Return ConfigParser object.
        return cls(config, run_id, verbosity)

    # Setting read-only attributes.
    @property
    def config(self):
        return self._config

    @property
    def result_dir(self):
        return self._result_dir

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def summary_dir(self):
        return self._summary_dir