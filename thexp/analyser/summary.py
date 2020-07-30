"""
TODO 该类下各个功能还需要思考是否保留以及用途
"""

import numpy as np

class ValueSummary():
    def __init__(self,name,values):
        self.name = name
        self.values = np.array(values)

    @property
    def min(self):
        return self.values.min()

    @property
    def max(self):
        return self.values.max()

    @property
    def mean(self):
        return self.values.mean()

    @property
    def std(self):
        return self.values.std()

    def to_attr(self):
        from ..base_classes.attr import attr
        res = attr()
        res['{}-MIN'.format(self.name)] = self.min
        res['{}-MAX'.format(self.name)] = self.max
        res['{}-MEAN'.format(self.name)] = self.mean
        res['{}-STD'.format(self.name)] = self.std
        return res