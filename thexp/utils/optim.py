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
import pprint as pp
from functools import partial

import torch
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.asgd import ASGD
from torch.optim.lbfgs import LBFGS
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop
from torch.optim.rprop import Rprop
from torch.optim.sgd import SGD
from torch.optim.sparse_adam import SparseAdam


class DeviceParam:
    def __init__(self, cpu_final=False):
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device("cuda:{}".format(i)))
        if cpu_final:
            devices.append(torch.device("cpu"))




class OptimParam:
    O_Adadelta = Adadelta
    O_Adagrad = Adagrad
    O_Adam = Adam
    O_AdamW = AdamW
    O_SparseAdam = SparseAdam
    O_Adamax = Adamax
    O_ASGD = ASGD
    O_LBFGS = LBFGS
    O_RMSprop = RMSprop
    O_Rprop = Rprop
    O_SGD = SGD

    def __init__(self):
        self.optim_param = dict()
        self.optim_dict = dict(
            adadelta=Adadelta,
            adagrad=Adagrad,
            adam=Adam,
            adamw=AdamW,
            sparseadam=SparseAdam,
            adamax=Adamax,
            asgd=ASGD,
            lbfgs=LBFGS,
            rmsprop=RMSprop,
            rprop=Rprop,
            sgd=SGD,
        )

    def choose(self, optimizer):
        if issubclass(optimizer, Optimizer):
            pass
        else:
            optimizer = self.optim_dict[optimizer.lower()]

    def create(self, optimizer, param):
        assert issubclass(optimizer, Optimizer)
        return optimizer(param, **self[optimizer.__name__.lower()])

    def __iter__(self):
        for k, v in self.optim_param.items():
            yield partial(self.optim_dict[k], v)

    def __setitem__(self, key, value):
        self.optim_param[key] = value

    def __getitem__(self, item):
        return self.optim_param[item]

    class OptimBuilder():
        def __init__(self, param):
            self.param = param  # type:(OptimParam)

        def finish(self):
            return self.param

        def Adadelta(self, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
            self.param["adadelta"] = dict(
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
            )
            return self

        def Adagrad(self, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
            self.param["adagrad"] = dict(
                lr=lr,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
            )
            return self

        def Adam(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
            self.param["adam"] = dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
            return self

        def AdamW(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
            self.param["adamw"] = dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
            return self

        def SparseAdam(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
            self.param["sparseadam"] = dict(
                lr=lr,
                betas=betas,
                eps=eps,
            )
            return self

        def Adamax(self, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
            self.param["adamax"] = dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            return self

        def ASGD(self, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
            self.param["asgd"] = dict(
                lr=lr,
                lambd=lambd,
                alpha=alpha,
                t0=t0,
                weight_decay=weight_decay,
            )
            return self

        def LBFGS(self, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09,
                  history_size=100, line_search_fn=None):
            self.param["lbfgs"] = dict(
                lr=lr,
                max_iter=max_iter,
                max_eval=max_eval,
                tolerance_grad=tolerance_grad,
                tolerance_change=tolerance_change,
                history_size=history_size,
                line_search_fn=line_search_fn,
            )
            return self

        def RMSprop(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
            self.param["rmsprop"] = dict(
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
            )
            return self

        def Rprop(self, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)):
            self.param["rprop"] = dict(
                lr=lr,
                etas=etas,
                step_sizes=step_sizes,
            )
            return self

        def SGD(self, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False):
            self.param["sgd"] = dict(
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
            return self

    @staticmethod
    def build():
        return OptimParam.OptimBuilder(OptimParam())

    def __repr__(self) -> str:
        return pp.pformat(self.optim_param)
