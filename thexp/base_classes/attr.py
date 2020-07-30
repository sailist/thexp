"""

"""
import copy
import torch
import numpy as np
from collections import OrderedDict

from typing import Any


class attr(OrderedDict):
    """
    和EasyDict类似，相比于EasyDict，更好的地方在于参数是有序的，可能会在某些情况下更方便一些
    """

    @staticmethod
    def __parse_value(v):
        if isinstance(v, attr):
            pass
        elif isinstance(v, dict):
            v = attr.from_dict(v)
        return v

    def __getattr__(self, item):
        if item not in self or self[item] is None:
            self[item] = attr()
        return self[item]

    def __setattr__(self, name: str, value: Any) -> None:
        value = self.__parse_value(value)
        self[name] = value

    def __getitem__(self, k):
        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            return super().__getitem__(ks[0])

        cur = self
        for tk in ks:
            cur = cur.__getitem__(tk)
        return cur

    def __setitem__(self, k, v):
        v = self.__parse_value(v)

        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            super().__setitem__(ks[0], v)
        else:
            cur = self
            for tk in ks[:-1]:
                cur = cur.__getattr__(tk)
            cur[ks[-1]] = v

    def __contains__(self, o: object) -> bool:
        try:
            _ = self[o]
            return True
        except:
            return False

    def __copy__(self):
        # res = attr()
        return self.from_dict(self)

    def walk(self):
        for k, v in self.items():
            if isinstance(v, attr):
                for ik, iv in v.walk():
                    ok = "{}.{}".format(k, ik)
                    yield ok, iv
            else:
                yield k, v

    def jsonify(self) -> dict:
        """
        获取可被json化的dict，目前仅支持 数字类型、字符串、bool、list/set 类型
        :return:
        """
        import numbers
        res = dict()
        for k, v in self.items():
            if isinstance(v, (numbers.Number, str, bool)):
                res[k] = v
            elif isinstance(v, (set, list)):
                v = [vv for vv in v if isinstance(vv, (numbers.Number, str, bool))]
                res[k] = v
            elif isinstance(v, attr):
                v = v.jsonify()
                res[k] = v

        return res

    def hash(self) -> str:
        from ..utils.paths import hash
        return hash(self)

    def copy(self):
        return self.__copy__()

    def replace(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v
        return self

    @classmethod
    def from_dict(cls, dic: dict):
        res = cls()
        for k, v in dic.items():
            if isinstance(v, attr):
                v = v.from_dict(v)
            elif isinstance(v, (list, tuple, torch.Tensor, np.ndarray)):
                v = cls._copy_iters(v)

            res[k] = v
        return res

    @staticmethod
    def _copy_iters(item):
        if isinstance(item, torch.Tensor):
            return item.clone()
        elif isinstance(item, np.ndarray):
            return item.copy()
        elif isinstance(item, list):
            return list(attr._copy_iters(i) for i in item)
        elif isinstance(item, tuple):
            return tuple(attr._copy_iters(i) for i in item)
        else:
            return copy.copy(item)
