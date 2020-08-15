# -*- coding: utf-8 -*-
from . import *
from . import optimizer
from . import scheduler

import tensorflow as tf

optimizer_dict = {'SGD': tf.keras.optimizers.SGD, 'Adadelta': tf.keras.optimizers.Adadelta,
                  'Adagrad': tf.keras.optimizers.Adagrad, 'Adam': tf.keras.optimizers.Adam,
                  'RMSprop': tf.keras.optimizers.RMSprop}
scheduler_dict = {'StepLR': tf.keras.optimizers.schedules.ExponentialDecay}
