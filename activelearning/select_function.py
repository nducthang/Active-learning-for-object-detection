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
        # probas có dạng {"tên ảnh": xác suất}
        return random.sample(probas.keys(), num_select)

class UncertaintySampling(BaseSelectFunction):
    @staticmethod
    def select(num_select, ):
        """
        docstring
        """
        pass
