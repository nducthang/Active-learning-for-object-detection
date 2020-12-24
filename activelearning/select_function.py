from collections import OrderedDict
import activelearning.config as config
import glob
import random

class BaseSelectFunction(object):
    def __init__(self):
        pass
    
    def select(self):
        pass


class RandomSelect(BaseSelectFunction):
    @staticmethod
    def select(num_select, probas):
        return random.sample(probas.keys(), num_select)

class EntropySelect(BaseSelectFunction):
    @staticmethod
    def select(num_select, files):
        pass

class MarginSamplingSelect(BaseSelectFunction):
    @staticmethod
    def select(num_select, files):
        pass