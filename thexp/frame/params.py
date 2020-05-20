"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""
import copy
import pprint as pp
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import fire

from ..utils.lazy import torch


class attr(OrderedDict):
    def __getattr__(self, item):
        if item not in self or self[item] is None:
            self[item] = attr()
        return self[item]

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = attr.from_dict(value)
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
        if isinstance(v, dict):
            v = attr.from_dict(v)

        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            super().__setitem__(ks[0], v)
        else:
            cur = self
            for tk in ks[:-1]:
                cur = cur.__getattr__(tk)
            cur[ks[-1]] = v

    @staticmethod
    def from_dict(dic: dict):
        res = attr()
        for k, v in dic.items():
            if isinstance(v, dict):
                v = attr.from_dict(v)
            res[k] = v
        return res

    def __copy__(self):
        return attr(
            **{k: copy.copy(v) for k, v in self.items()}
        )

    def hash(self) -> str:
        from ..utils.generel_util import hash
        return hash(self)

    def jsonify(self):
        """
        获取可被json化的dict，其中包含的任意
        :return:
        """



class BoundCheckError(BaseException):
    pass


class NewParamWarning(Warning):
    pass


class BaseParams:
    """TODO 将build_exp_name 的前缀单独放，然后参数放子目录"""

    def __init__(self):
        self._param_dict = attr()
        self._exp_name = None
        self._bound = {}

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __contains__(self, item):
        return item in self._param_dict

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            res = self._check(name, value)
            if res is not None and not res:
                raise BoundCheckError("param '{}' checked failed.".format(name))
            self._param_dict[name] = value

    def __getattr__(self, item):
        # if item not in self._param_dict:
        #     raise AttributeError(item)
        return self._param_dict.__getattr__(item)

    def __repr__(self):
        return "{}".format(self.__class__.__name__) + pp.pformat([(k, v) for k, v in self._param_dict.items()])

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            self._param_dict.pop(name)

    def __delitem__(self, key):
        key = str(key)
        self.__delattr__(key)

    def _check(self, name, value):
        if name in self._bound:
            self._bound[name](value)

    def arange(self, k, default, left=float("-inf"), right=float("inf")):
        """
        约束 key 的选择范围（连续）

        如果有多段连续，请使用 lambda_bound() 方法
        :param k: 要约束的键
        :param default: 默认值
        :param left:
        :param right:
        :return:
        """

        def check(x):
            if x < left or x > right:
                raise BoundCheckError("param '{}' must be within range({}, {})".format(k, left, right))

        self._bound[k] = check
        self[k] = default

    def choice(self, k, *choices):
        """
        约束 key 的选择范围（离散）
        :param k: 要约束的键
        :param choices: 要选择的常量，第一个将作为默认值
        :return:
        """
        def check(x):
            if x not in choices:
                raise BoundCheckError("param '{}' is enum of {}".format(k, choices))

        self._bound[k] = check
        self[k] = choices[0]

    def lambda_bound(self, k, default, check_lmd):
        """
        约束 key 符合 check_lmd 的要求
        :param k:
        :param default:
        :param check_lmd: 一个函数，接受一个参数，为 key 接受新值时的 value
            该函数若返回None，则表明异常会在检查过程中抛出，若返回值为任意可以判断真值的类型，则根据真值类型判断
                为假则抛出 BoundCheckError
        :return:
        """
        self._bound[k] = check_lmd
        self[k] = default

    def grid_search(self, k, iterable: Iterable):
        snap = copy.copy(self._param_dict)
        for v in iterable:
            for sk, sv in snap.items():
                self[sk] = sv
            self[k] = v
            yield self

    def from_args(self):
        def func(**kwargs):
            for k, v in kwargs.items():
                try:
                    self[k]
                except:
                    warnings.simplefilter('always', NewParamWarning)
                    warnings.warn(
                        "{} is a new param,please check your spelling.\n it's more recommended to define in advance.".format(
                            k))
                self[k] = v

        fire.Fire(func)

    def items(self):
        return self._param_dict.items()

    def keys(self):
        for k in self._param_dict:
            yield k

    def update(self, items):
        """
        :param items:
        :return:
        """
        for k, v in items.items():
            self._param_dict[k] = v

    def hash(self):
        return self._param_dict.hash()

    def __eq__(self, other):
        if isinstance(other, BaseParams):
            return self.hash() == other.hash()
        return False


class Params(BaseParams):
    Attr = attr

    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.eidx = 1
        self.idx = 0
        self.global_step = 0
        self.device = "cuda:0" if torch().cuda.is_available() else "cpu"
        self.dataset = None
        self.architecture = None
        self.optim = None


if __name__ == '__main__':
    pass
