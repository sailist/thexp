"""
TODO 该类下各个功能还需要思考是否保留以及用途
"""
import operator as op
import numpy as np


class ConstrainBuilder():
    def __getattr__(self, item):
        return Constrain(item)


class Constrain():
    def __init__(self, name):
        self._name = name
        self._constrain = None
        self._value = None

    def __lt__(self, other):
        self._constrain = op.lt
        self._value = other
        return self

    def lt(self, other):
        self._constrain = op.lt
        self._value = other
        return self

    def __gt__(self, other):
        self._constrain = op.gt
        self._value = other
        return self

    def gt(self, other):
        self._constrain = op.gt
        self._value = other
        return self

    def __le__(self, other):
        self._constrain = op.le
        self._value = other
        return self

    def le(self, other):
        self._constrain = op.le
        self._value = other
        return self

    def __ge__(self, other):
        self._constrain = op.ge
        self._value = other
        return self

    def ge(self, other):
        self._constrain = op.ge
        self._value = other
        return self

    def __eq__(self, other):
        self._constrain = op.eq
        self._value = other
        return self

    def eq(self, other):
        self._constrain = op.eq
        self._value = other
        return self

    def __ne__(self, other):
        self._constrain = op.ne
        self._value = other
        return self

    def ne(self, other):
        self._constrain = op.ne
        self._value = other
        return self

    @property
    def max(self):
        self.constrain = np.max
        return self

    @property
    def min(self):
        self.constrain = np.min
        return self
