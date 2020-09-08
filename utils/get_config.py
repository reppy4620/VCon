import os
import json
import yaml


class Map(dict):
    """
    dict-like class for accessing dict values as accessing class attributes

    e.g.
    >> d = {'a': 1, 'b': 2, 'c': {'aa': 11, 'bb': 22}}
    >> map = Map(d)
    >> map.a
    1
    >> map.c.aa
    11
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    # if child is dict object, change to Map object recursively
                    if isinstance(v, dict):
                        v = Map(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        return Map(state)


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


def construct_odict(loader, node):
    return Map(loader.construct_pairs(node))


yaml.add_representer(Map, represent_odict)
yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)


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
    config = Map(_config)
    return config
