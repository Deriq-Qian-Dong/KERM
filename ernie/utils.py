import json
import six
import os
class HParams(object):
    """Hyper paramerter"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise ValueError('key(%s) not in HParams.' % key)
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.to_dict())

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    @classmethod
    def from_json(cls, json_str):
        """doc"""
        d = json.loads(json_str)
        if type(d) != dict:
            raise ValueError('json object must be dict.')
        return HParams.from_dict(d)

    def get(self, key, default=None):
        """doc"""
        return self.__dict__.get(key, default)

    @classmethod
    def from_dict(cls, d):
        """doc"""
        if type(d) != dict:
            raise ValueError('input must be dict.')
        hp = HParams(**d)
        return hp

    def to_json(self):
        """doc"""
        return json.dumps(self.__dict__)

    def to_dict(self):
        """doc"""
        return self.__dict__
    
    def print_config(self):
        for key,value in self.__dict__.items():
            print(key+":",value)

    def join(self, other):
        """doc"""
        if not isinstance(other, HParams):
            raise ValueError('input must be HParams instance.')
        self.__dict__.update(**other.__dict__)
        return self

def _get_dict_from_environ_or_json_or_file(args, env_name):
    if args == '':
        return None
    if args is None:
        s = os.environ.get(env_name)
    else:
        s = args
        if os.path.exists(s):
            s = open(s).read()
    if isinstance(s, six.string_types):
        try:
            r = eval(s)
        except SyntaxError as e:
            raise ValueError('json parse error: %s \n>Got json: %s' %
                             (repr(e), s))
        return r
    else:
        return s  #None


def parse_file(filename):
    """useless api"""
    d = _get_dict_from_environ_or_json_or_file(filename, None)
    if d is None:
        raise ValueError('file(%s) not found' % filename)
    return d