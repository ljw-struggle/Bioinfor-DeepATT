# -*- coding: utf-8 -*-
from . import *
from .loader import *
from .trainer import *

loader_dict = {'SequenceLoader': SequenceLoader}
trainer_dict = {'EarlyStopTrainer': EarlyStopTrainer}
