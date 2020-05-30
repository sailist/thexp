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
import json
import os
import pprint as pp
import warnings
from collections import defaultdict

import fire
from collections.abc import Iterable
from typing import Any

from ..base_classes.attr import attr
from ..base_classes.errors import BoundCheckError, NewParamWarning
from ..utils.lazy import torch


class BaseParams:
    """TODO 将build_exp_name 的前缀单独放，然后参数放子目录"""

    def __init__(self):
        self._param_dict = attr()
        self._exp_name = None
        self._bound = {}
        self._bind = defaultdict(list)

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
            if name in self._bind:
                for _v, _bind_k, _bind_v in self._bind[name]:
                    if _v == value:
                        self.__setattr__(_bind_k, _bind_v)

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
        Args:
            k: 要约束的键
            default: 默认值
            left:
            right:

        Returns:

        """

        def check(x):
            if x < left or x > right:
                raise BoundCheckError("param '{}' must be within range({}, {})".format(k, left, right))

        self._bound[k] = check
        self[k] = default

    def choice(self, k, *choices):
        """
        约束 key 的选择范围（离散）

        Args:
            k: 要约束的键
            *choices: 要选择的常量，第一个将作为默认值

        Returns:

        """

        def check(x):
            if x not in choices:
                raise BoundCheckError("param '{}' is enum of {}".format(k, choices))

        self._bound[k] = check
        self[k] = choices[0]

    def bind(self, k, v, bind_k, bind_v):
        """
        k 变化到 v 后，让相应的 bind_k 的值变化到 bind_v
        """
        self._bind[k].append((v, bind_k, bind_v))

    def lambda_bound(self, k, default, check_lmd):
        """
        约束 key 符合 check_lmd 的要求
        Args:
            k:
            default:
            check_lmd: 一个函数，接受一个参数，为 key 接受新值时的 value
            该函数若返回None，则表明异常会在检查过程中抛出，若返回值为任意可以判断真值的类型，则根据真值类型判断
                为假则抛出 BoundCheckError

        Returns:

        """
        self._bound[k] = check_lmd
        self[k] = default

    def grid_search(self, key, iterable: Iterable):
        """

        Args:
            key:
            iterable:

        Returns:

        """
        snapshot = copy.copy(self._param_dict)
        for v in iterable:
            for sk, sv in snapshot.items():
                self[sk] = sv
            self[key] = v
            yield self

    def from_args(self):
        """
        从命令行参数中设置参数值
        Returns:

        """

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
        return self

    def from_json(self, fn):
        """
        从 json 中获取参数值
        Args:
            fn:

        Returns:

        """
        if os.path.exists(fn):
            with open(fn, encoding='utf-8') as r:
                res = json.load(r)
                for k, v in res.items():
                    self[k] = v
        return self

    def to_json(self, fn: str):
        """
        以json格式保存文件，注意，该方法保存的json内容基于 attr 的jsonify() 方法，不可序列化的格式无法被保存
        Args:
            fn:

        Returns:

        """
        with open(fn, 'w', encoding='utf-8') as w:
            json.dump(self.inner_dict.jsonify(), w, indent=2)

    def items(self):
        return self._param_dict.items()

    def keys(self):
        for k in self._param_dict:
            yield k

    def update(self, dic: dict):
        """

        Args:
            dic:

        Returns:

        """
        for k, v in dic.items():
            self._param_dict[k] = v

    def hash(self) -> str:
        """
        返回对参数的定义顺序及其相应值的一个hash，理论上，两个Param 对象的hash方法返回的参数相同，
        则两者具有相同的参数和参数值

        Returns:

        """
        return self._param_dict.hash()

    @property
    def inner_dict(self) -> attr:
        return self._param_dict

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
        self.ignore_set = set()


if __name__ == '__main__':
    pass
