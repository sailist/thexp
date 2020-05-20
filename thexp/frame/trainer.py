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
import bisect
import pprint as pp
import warnings
from functools import lru_cache
from functools import wraps
from typing import Any

from .databundler import DataBundler
from .meter import AvgMeter
from .params import Params
from .saver import Saver
from ..base_classes.metaclasses import Merge
from ..globals import _BUILTIN_PLUGIN
from ..utils.lazy import torch, np


def build_exp_name(trainer_name):
    return trainer_name.lower().replace("trainer", "Exp")


class BaseTrainer(metaclass=Merge):
    __exp_name__ = None
    _call_backs = {
        "train", "train_epoch", "train_step", "test", "eval", "train_on_batch",
        "regist_databundler", "train_batch", "test_eval_logic", "predict",
        "load_keypoint", "load_checkpoint", "load_model", "save_keypoint", "save_checkpoint", "save_model",
    }

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if cls.__exp_name__ is None:
            cls.__exp_name__ = build_exp_name(cls.__name__)

        def wrapper(func, _call_set: list):
            @wraps(func)
            def _newfunc(*args, **kwargs):
                for callback in _call_set:
                    if callback.enable:
                        callback.on_begin(self, func, self.params, *args, **kwargs)
                try:
                    _meter = func(*args, **kwargs)
                except BaseException as e:
                    _handles = [callback.on_exception(self, func, self.params, e, *args, **kwargs)
                                for callback in _call_set]

                    if any(_handles):
                        return None
                    else:
                        raise e

                for callback in _call_set:
                    if callback.enable:
                        callback.on_end(self, func, self.params, _meter, *args, **kwargs)
                return _meter

            return _newfunc

        self._callback_set = []
        self._callback_name_set = set()

        vars = dir(self)
        for name in vars:
            if name not in self._call_backs:
                continue
            if name.startswith("_"):
                continue
            value = getattr(self, name, None)
            if value is None:
                continue
            if callable(value):
                setattr(self, name, wrapper(value, self._callback_set))
        return self

    def __init__(self, params: Params = None):
        self._model_dict = {}
        self._optim_dict = {}
        self._other_state_dict = {}
        self._vector_dict = {}
        self._checkpoint_plug = {}
        self._databundler_dict = {}
        self.train_epoch_toggle = False
        self.train_toggle = False
        if params is not None:
            self.params = params
            self.device = torch().device(params.device)
        self.initial()

    def initial(self):
        """
        will call:
            self.initial_trainer(self.params)
            self.initial_models(self.params)
            self.initial_datasets(self.params)
            self.initial_callback()

        :param exps_dir:
        :return:
        """
        from .experiment import Experiment
        from ..utils.gitutils import locate_cls
        pre = self.__class__.__module__.split('.')[-1]
        self.experiment = Experiment("{}.{}".format(self.__exp_name__, pre))
        # self.params TODO
        # self.experiment.makedir('params')
        self.experiment.regist_plugin(_BUILTIN_PLUGIN.trainer, {
            'param_hash': self.params.hash(),
            'path': __file__,
            'loaction': locate_cls(self.__class__),
            'module': self.__class__.__module__,
            'class': self.__class__.__name__
        })
        self.experiment.start()

        # self.reporter = Reporter(self.experiment.makedir("plot"))
        self.models(self.params)
        self.datasets(self.params)
        self.callbacks(self.params)

    def _regist_databundler(self, key, val):
        from torch.utils.data.dataloader import DataLoader
        assert isinstance(val, (DataBundler, DataLoader))
        if isinstance(val, DataLoader):
            val = DataBundler().add(val)
        self._databundler_dict[key] = val

    def regist_databundler(self, train=None, eval=None, test=None):
        if train is not None:
            self._regist_databundler("train", train)
        if eval is not None:
            self._regist_databundler("eval", eval)
        if test is not None:
            self._regist_databundler("tests", test)

    def train(self):
        params = self.params
        for eidx in range(params.eidx, params.epoch + 1):
            self.train_epoch(params.eidx, params)
            params.eidx = eidx + 1
            if self.train_toggle:
                self.train_toggle = False
                break
        self.experiment.end()

    def train_epoch(self, eidx, params):
        avg = AvgMeter()
        for idx, batch_data in enumerate(self.train_dataloader):
            self.change_mode(True)
            meter = self.train_batch(eidx, idx, self.params.global_step, batch_data, params, self.device)
            avg.update(meter)
            self.change_mode(False)

            params.global_step += 1
            params.idx = idx
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

        return avg

    def train_step(self, steps):
        param = self.params
        i = 0
        avg = AvgMeter()
        while steps > 0:
            avg = AvgMeter()
            for idx, data in enumerate(self.train_dataloader):
                meter = self.train_batch(0, idx, i, data, param, self.device)
                steps -= 1
                avg.update(meter)
                if steps <= 0:
                    return avg
        return avg

    def feed_batchdata(self, batch_data):
        self.train_batch(0, 0, 0, batch_data, self.params, self.device)

    def test(self):
        loader = self.test_dataloader
        if loader is None:
            self.logger.info("Have no test dataset, ignored test.")
            return None
        return self.test_eval_logic(loader, self.params)

    def eval(self):
        loader = self.eval_dataloader
        if loader is None:
            self.logger.info("Have no eval dataset, ignored eval.")
            return None
        return self.test_eval_logic(loader, self.params)

    @property
    @lru_cache()
    def writer(self):
        from torch.utils.tensorboard import SummaryWriter
        from ..globals import _PLUGIN_WRITER
        d = self.experiment.makedir(_PLUGIN_WRITER.dir_name)
        kwargs = {
            _PLUGIN_WRITER.log_dir: d,
            _PLUGIN_WRITER.filename_suffix: '.bd'
        }
        self.experiment.regist_plugin(_BUILTIN_PLUGIN.writer, kwargs)

        res = SummaryWriter(**kwargs)

        def close():
            res.flush()
            res.close()

        self.experiment.regist_exit_hook(close)
        return res

    @property
    @lru_cache()
    def logger(self):
        from .logger import Logger
        logger = Logger()
        fn = logger.add_log_dir(self.experiment.test_dir)
        self.experiment.regist_plugin(_BUILTIN_PLUGIN.logger, dict(
            log_dir=self.experiment.test_dir,
            fn=fn,
        ))
        return logger

    @property
    @lru_cache()
    def saver(self):
        d = self.experiment.makedir("modules")
        kwargs = dict(
            max_to_keep=3,
            ckpt_dir=d
        )
        self.experiment.regist_plugin(_BUILTIN_PLUGIN.saver, kwargs)
        return Saver(**kwargs)

    @property
    @lru_cache()
    def rnd(self):
        from .rndmanager import RndManager
        d = self.experiment.make_exp_dir("rnd")
        kwargs = dict(
            save_dir=d,
        )
        self.experiment.regist_plugin(_BUILTIN_PLUGIN.rnd, kwargs)
        return RndManager(**kwargs)

    @property
    def model_dict(self):
        return self._model_dict

    @property
    def optimizer_dict(self):
        return self._optim_dict

    @property
    def train_dataloader(self) -> DataBundler:
        return self._databundler_dict.get("train", None)

    @property
    def eval_dataloader(self) -> DataBundler:
        return self._databundler_dict.get("eval", None)

    @property
    def test_dataloader(self) -> DataBundler:
        return self._databundler_dict.get("tests", None)

    @classmethod
    def from_params(cls, params: Params = None):
        return cls(params)

    def regist_checkpoint(self, key, func):
        self._checkpoint_plug[key] = func

    def save_keypoint(self, extra_info=None, replacement=False):
        state_dict = self.checkpoint_dict()
        fn = self.saver.save_keypoint(state_dict["_eidx"], state_dict, extra_info, replacement)
        self.logger.info("save keypoint in {}".format(fn))
        return fn

    def save_checkpoint(self, extra_info=None, replacement=False):
        state_dict = self.checkpoint_dict()
        fn = self.saver.save_checkpoint(state_dict["_eidx"], state_dict, extra_info, replacement)
        self.logger.info("save checkpoint in {}".format(fn))
        return fn

    def save_model(self, extra_info=None):
        state_dict = self.model_state_dict()
        fn = self.saver.save_model(self.params.eidx, state_dict, extra_info)
        self.logger.info("save model in {}".format(fn))
        return fn

    def add_callback(self, callback):
        """
        添加一个回调函数
        :type callable,str
        :param callback:
        :return:
        """
        msg = None
        cb_name = callback.__class__.__name__
        if callback not in self._callback_set and cb_name in self._callback_name_set:
            msg = "Callback duplicate."
            callback.on_hook_failed(self, msg)

        if msg is not None:
            return False
        bisect.insort(self._callback_set, callback)
        self._callback_name_set.add(cb_name)

        callback._trainer = self
        callback.on_hooked(self, self.params)
        self.logger.info("{} hooked on {}.".format(callback, self))
        return True

    def reload_callback(self, callback):
        self.remove_callback(callback.__class__)
        return self.add_callback(callback)

    def remove_callback(self, callback):
        """

        :param callback: str / class / callback instance
        :return:
        """
        msg = None
        from .callbacks import BaseCallback

        if issubclass(callback, BaseCallback):
            for cb in self._callback_set:
                if cb.__class__.__name__ == callback.__name__:
                    callback = cb
                    break

        if isinstance(callback, str):
            for cb in self._callback_set:
                if cb.__class__.__name__ == callback:
                    callback = cb
                    break

        if callback not in self._callback_set:
            return False

        cb_name = callback.__class__.__name__
        self._callback_set.remove(callback)
        self._callback_name_set.remove(cb_name)
        self.logger.info("{} unhooked from {}.".format(callback, self))
        return True

    '''module和optim的一部分方法集成'''

    def load_latast_checkpoint(self):
        ckpt, info = self.saver.load_latest_checkpoint()
        self.load_checkpoint_dict(ckpt)
        self.logger.raw(pp.pformat(info))

    def load_checkpoint(self, fn):
        ckpt, info = self.saver.load_state_dict(fn)
        self.load_checkpoint_dict(ckpt)
        self.logger.raw(pp.pformat(info))

    def load_model(self, fn, strict=True):
        ckpt, info = self.saver.load_state_dict(fn)
        self.load_model_state_dict(ckpt, strict=strict)
        self.logger.raw(pp.pformat(info))

    def load_checkpoint_dict(self, state_dict):
        self.logger.raw("loading checkpoint")
        self.params.eidx = state_dict['eidx']
        self.params.idx = state_dict['idx']
        self.params.global_step = state_dict['global_step']
        self.load_model_state_dict(state_dict["model"])
        self.load_optim_state_dict(state_dict["optim"])
        self.load_other_state_dict(state_dict["other"])
        self.load_vector_dict(state_dict["vector"])
        self.load_plug_state_dict(state_dict['plug'])

    def load_model_state_dict(self, state_dict, strict=True):
        self.logger.inline("loading model: ", append=True)
        for k in self._model_dict:
            self.logger.raw(k)
            if k in state_dict:
                self._model_dict[k].load_state_dict(state_dict[k], strict=strict)
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_optim_state_dict(self, state_dict, strict=False):
        self.logger.inline("loading optimizers: ", append=True)
        for k in self._optim_dict:
            self.logger.raw(k)
            if k in state_dict:
                self._optim_dict[k].load_state_dict(state_dict[k])
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_other_state_dict(self, state_dict, strict=False):
        self.logger.inline("loading other: ", append=True)
        for k in self._other_state_dict:
            self.logger.raw(k)
            if k in state_dict:
                self._other_state_dict[k].load_state_dict(state_dict[k])
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_vector_dict(self, state_dict, strict=False):
        self.logger.inline("loading vectors: ", append=True)
        for k in self._vector_dict:
            self.logger.raw(k)
            if k in state_dict:
                self.__setattr__(k, state_dict[k])
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_plug_state_dict(self, state_dict, strict=False):
        self.logger.inline("loading plugs: ", append=True)
        for k in self._checkpoint_plug:
            self.logger.raw(k)
            if k in state_dict:
                try:
                    self._checkpoint_plug[k](self, state_dict[k], strict)
                except:
                    if strict:
                        raise KeyError("{} not match this load function".format(k))
                    else:
                        warnings.warn("{} not match this load function".format(k))
            else:
                if strict:
                    raise KeyError(k)
                else:
                    warnings.warn("{} not found in state_dict".format(k))
        self.logger.newline()

    def load_state_dict(self, strict=True, **kwargs):
        for k, v in kwargs.items():
            if k in self._model_dict:
                self._model_dict[k].load_state_dict(v, strict)
            elif k in self._optim_dict:
                self._optim_dict[k].load_state_dict(v)
            elif k in self._other_state_dict:
                self._other_state_dict[k].load_state_dict(v)
            elif k in self._vector_dict:
                self.__setattr__(k, v)
            elif k in self._checkpoint_plug:
                self._checkpoint_plug[k](self, v, strict)
            elif strict:
                raise KeyError(k)
            else:
                warnings.warn("{} not found in all state_dict".format(k))

    def estimate_memory(self):
        for _, v in self._model_dict.items():
            pass

    def checkpoint_dict(self):
        val = dict(
            model=self.model_state_dict(),
            optim=self.optim_state_dict(),
            other=self.other_state_dict(),
            vector=self.vector_state_dict(),
            plug=self.plug_state_dict(),
            eidx=self.params.eidx,
            idx=self.params.idx,
            global_step=self.params.global_step,
        )
        return val

    def model_state_dict(self):
        return {k: v.state_dict() for k, v in self._model_dict.items()}

    def optim_state_dict(self):
        return {k: v.state_dict() for k, v in self._optim_dict.items()}

    def other_state_dict(self):
        return {k: v.state_dict() for k, v in self._other_state_dict.items()}

    def vector_state_dict(self):
        return {k: v for k, v in self._vector_dict.items()}

    def plug_state_dict(self):
        return {k: getattr(self, k, None) for k in self._checkpoint_plug}

    def change_mode(self, train=True):
        for k, v in self._model_dict.items():
            if train:
                v.train()
            else:
                v.eval()

    def to(self, device):
        for k, v in self._model_dict.items():
            self.__setattr__(k, v.to(device))

    '''magic functions'''

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        from torch.optim.optimizer import Optimizer
        if isinstance(value, torch().device):
            pass
        elif isinstance(value, torch().nn.Module):
            self._model_dict[name] = value
        elif isinstance(value, Optimizer):
            self._optim_dict[name] = value
        elif isinstance(value, (torch().Tensor, np().ndarray)):
            self._vector_dict[name] = value
        elif hasattr(value, "state_dict"):
            self._other_state_dict[name] = value

    # need to reimplement

    def train_batch(self, eidx, idx, global_step, batch_data, params: Params, device: torch().device):
        raise NotImplementedError()

    def test_eval_logic(self, dataloader, param: Params):
        raise NotImplementedError()

    def predict(self, xs):
        raise NotImplementedError()

    def callbacks(self, params: Params):
        pass

    def datasets(self, params: Params):
        pass

    def models(self, params: Params):
        pass


class Trainer(BaseTrainer):

    def callbacks(self, params: Params):
        super().callbacks(params)

    def datasets(self, params: Params):
        super().datasets(params)

    def models(self, params: Params):
        super().models(params)
