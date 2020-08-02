"""

"""
import copy
import json
import os
import pprint as pp
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, overload

import fire
import torch

from ..base_classes.attr import attr
from ..base_classes.errors import BoundCheckError, NewParamWarning
from ..base_classes.params_vars import ParamsFactory, OptimParams
from ..utils.environ import ENVIRON_


class BaseParams:
    ENV = ENVIRON_

    def __init__(self):
        self._param_dict = attr()
        self._copy = attr()
        self._exp_name = None
        self._bound = {}
        self._lock = False
        self._bind = defaultdict(list)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __contains__(self, item):
        return item in self._param_dict

    def __setattr__(self, name: str, value: Any) -> None:
        from ..base_classes.defaults import default
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            res = self._check(name, value)
            if res is not None and not res:
                raise BoundCheckError("param '{}' checked failed.".format(name))

            if isinstance(value, default):  # 设置默认值
                if name not in self._param_dict:
                    if value.warn:
                        warnings.warn(
                            "'{}' is a new param,please check your spelling. It's more recommended to define in advance.".format(
                                name))
                    self._param_dict[name] = value.default
            else:
                self._param_dict[name] = value
            if name in self._bind:
                for _v, _bind_k, _bind_v in self._bind[name]:
                    if _v == value:
                        self.__setattr__(_bind_k, _bind_v)

    def __getattr__(self, item):
        if self._lock:
            if item not in self._param_dict:
                raise AttributeError(item)

        return self._param_dict.__getattr__(item)

    def __repr__(self):
        return "{}".format(self.__class__.__name__) + pp.pformat([(k, v) for k, v in self._param_dict.items()])

    __str__ = __repr__

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            self._param_dict.pop(name)

    def __delitem__(self, key):
        key = str(key)
        self.__delattr__(key)

    def __eq__(self, other):
        if isinstance(other, BaseParams):
            return self.hash() == other.hash()
        return False

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
        return default

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
        return choices[0]

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
        return default

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

    def grid_range(self, count):
        snapshot = copy.copy(self._param_dict)
        for i in range(count):
            for sk, sv in snapshot.items():
                self[sk] = sv
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
                        "'{}' is a new param,please check your spelling.\n it's more recommended to define in advance.".format(
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

        return self

    def hash(self) -> str:
        """
        返回对参数的定义顺序及其相应值的一个hash，理论上，两个Param 对象的hash方法返回的参数相同，
        则两者具有相同的参数和参数值

        Returns:

        """
        return self._param_dict.hash()

    def lock(self):
        """
        锁定当前配置，如果当前配置未 lock，那么当尝试获取未分配的参数时候，会返回一个空的字典
        如果锁定，则在尝试获取未分配参数时，会抛出 AttributeError(key)
        Returns:

        """
        self._lock = True

    @property
    def inner_dict(self) -> attr:
        return self._param_dict

    def get(self, k, default=None):
        """
        获取某值，如果不存在，则返回默认值
        Args:
            k:
            default:

        Returns:

        """
        if k in self:
            return self[k]
        else:
            return default

    @staticmethod
    def default(value: Any = None, warn=False):
        """
        默认值，分配值时，仅当当前key没有分配时，分配该值作为键值。否则，该值会被忽略

        Examples:
        >>> params.margin = params.default(0.5,True)
        >>> params.margin = params.default(0.3,True)
        >>> print(params.margin)

        Args:
            value: 要设置的值
            warn: 当设置默认值时，抛出警告

        Returns:
            default(value, warn)
        """
        from ..base_classes.defaults import default
        return default(value, warn)

    def create_optim(self, **kwargs):
        return ParamsFactory.create_optim(**kwargs)

    def create_schedule(self, schedule_type, start, end, **kwargs):
        return ParamsFactory

    def replace(self, **kwargs):
        self.update(kwargs)
        return self

    @overload
    def create_optim(self, name='SGD', lr=None, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        pass

    @overload
    def create_optim(self, name='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        pass

    @overload
    def create_optim(self, name='Adadelta', lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        pass

    @overload
    def create_optim(self, name='Adagrad', lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        pass

    @overload
    def create_optim(self, name='AdamW', lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        pass

    @overload
    def create_optim(self, name='AdamW',
                     lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        pass

    @overload
    def create_optim(self, name='ASGD',
                     lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        pass

    @overload
    def create_optim(self, name='LBFGS',
                     lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
                     line_search_fn=None):
        pass

    @overload
    def create_optim(self, name='RMSprop', lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        pass

    @overload
    def create_optim(self, name='Rprop', lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        pass

    @overload
    def create_optim(self, name='SparseAdam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        pass


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
        self.optim = None  # type:OptimParams

    def dataloader(self):
        pass

    def optimizer(self):
        return self.optim.args


if __name__ == '__main__':
    Params().create_optim()

    from torch.optim.sgd import SGD
    from torch.optim.sparse_adam import SparseAdam
    from torch.optim import adam, adamw, adamax, adagrad, adadelta, asgd, sparse_adam, sgd, lr_scheduler, lbfgs, \
        optimizer, rmsprop, rprop
