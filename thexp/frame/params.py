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
from collections.abc import Iterable
import pprint as pp
import warnings
from collections import OrderedDict
from typing import Any

import fire
import torch


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
            self.check(name, value)
            self._param_dict[name] = value

    def __getattr__(self, item):
        if item not in self._param_dict:
            raise AttributeError(item)
        return self._param_dict[item]

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

    def check(self, name, value):
        if name in self._bound:
            self._bound[name](value)

    def arange(self, k, default, left=float("-inf"), right=float("inf")):
        """
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
        :param k: 要约束的键
        :param choices: 要选择的常量，第一个将作为默认值
        :return:
        """

        def check(x):
            if x not in choices:
                raise BoundCheckError("param '{}' is enum of {}".format(k, choices))

        self._bound[k] = check
        self[k] = choices[0]

    def grid_search(self,k,iterable:Iterable):
        for v in iterable:
            self[k] = v
            yield self

    # TODO  获取试验目录是否应该在Params类中获取？思考一下
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

    # @deprecated
    def _can_in_dir_name(self, obj):
        for i in [int, float, str, bool]:
            if isinstance(obj, i):
                return True
        if isinstance(obj, torch.Tensor):
            if len(obj.shape) == 0:
                return True
        return False

    def build_exp_name(self, *names, prefix="", sep="_", ignore_mode="add"):
        prefix = prefix.strip()
        res = []
        if len(prefix) != 0:
            res.append(prefix)
        if ignore_mode == "add":
            for name in names:
                if hasattr(self, name):
                    obj = getattr(self, name)
                    if self._can_in_dir_name(obj):
                        res.append("{}={}".format(name, obj))
                else:
                    res.append(name)

        elif ignore_mode == "del":
            for name in names:
                if hasattr(self, name):
                    obj = getattr(self, name)
                    if self._can_in_dir_name(obj):
                        res.append("{}={}".format(name, obj))
        else:
            assert False

        self._exp_name = sep.join(res)
        return self._exp_name

    def get_exp_name(self):
        assert self._exp_name is not None, "run build_exp_name() before get_exp_name()"
        return self._exp_name

    def update(self,items):
        """
        :param items:
        :return:
        """
        for k,v in items.items():
            self._param_dict[k] = v



class Params(BaseParams):
    Attr = attr
    def __init__(self):
        super().__init__()
        self.epoch = 10
        self.eidx = 1
        self.idx = 0
        self.global_step = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset = None
        self.architecture = None
        self.optim = None


if __name__ == '__main__':
    pass
