import json


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


# load config file from json as dict which can be accessed like config.param
def get_config(filename):
    with open(filename, 'r') as f:
        _config = json.load(f)
    config = Map(_config)
    return config
