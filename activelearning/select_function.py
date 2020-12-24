class BaseSelectFunction(object):
    def __init__(self):
        pass
    
    def select(self):
        pass


class RandomSelect(BaseSelectFunction):
    @staticmethod
    def select(probas_val):
        pass

class EntropySelect(BaseSelectFunction):
    @staticmethod
    def select(probas_val):
        pass

class MarginSamplingSelect(BaseSelectFunction):
    @staticmethod
    def select(probas_val):
        pass