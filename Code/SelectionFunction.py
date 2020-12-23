from numpy.lib.function_base import select
from sklearn.utils import check_random_state
import numpy as np


class BaseSelectionFunction(object):
    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        random_state = check_random_state(0)
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)
        return selection

class EntropySelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        e = (-probas_val*np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
        return selection

class MarginSamplingSelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        rev = np.sort(probas_val, axis=1)[:,::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:initial_labeled_samples]
        return selection