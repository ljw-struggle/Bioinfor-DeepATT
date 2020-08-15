# -*- coding: utf-8 -*-
import os
import csv
import json
import matplotlib.pyplot as plt

from datetime import datetime
from collections import OrderedDict

class Timer(object):
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

    @classmethod
    def now(cls):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S %p')

def create_dirs(dirs):
    """ Create dirs. (recurrent) """
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)

def write2txt(content, file_path):
    """ Write array to .txt file. """
    with open(file_path, 'w+') as f:
        for item in content:
            f.write(item + '\n')

def write2csv(content, file_path):
    """ Write array to .csv file. """
    with open(file_path, 'w+', newline='') as f:
        csv_writer = csv.writer(f, dialect='excel')
        for item in content:
            csv_writer.writerow(item)

def read_json(file_path):
    """ Read json to dict. """
    with open(file_path, 'rt') as f:
        return json.load(f, object_hook=OrderedDict)

def write_json(content, file_path):
    """ Write dict to json file. """
    with open(file_path, 'wt') as f:
        json.dump(content, f, indent=4, sort_keys=False)