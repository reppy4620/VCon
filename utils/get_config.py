import os
import json
import yaml

from .attribute_dict import AttributeDict


# Load config file from json or yaml which can be accessed like attributes of class
# This project basically uses yaml.
def get_config(filename):
    _, ext = os.path.splitext(filename)
    with open(filename, 'r') as f:
        if ext == '.json':
            _config = json.load(f)
        elif ext == '.yaml':
            _config = yaml.load(f, Loader=yaml.CLoader)
        else:
            raise ValueError('file extension must be .json or .yaml')
    config = AttributeDict(_config)
    return config
