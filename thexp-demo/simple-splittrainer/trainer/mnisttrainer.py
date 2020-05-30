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
from config import params
from thexp import *
import torch
from torch import nn
from torch.nn import functional as F


class MNISTTrainer(Trainer, callbacks.TrainCallback):
    __exp_name__ = "MNISTDemo"

    def callbacks(self, params: Params):
        callbacks.LoggerCallback().hook(self)
        callbacks.AutoRecord().hook(self)
        callbacks.EvalCallback(test_per_epoch=5).hook(self)

    def datasets(self, params: Params):
        from torch.utils.data.dataloader import DataLoader
        from thexp.torch.data.collate import AutoCollate
        from datasets import MNIST, FMNIST
        if params.dataset == 'mnist':
            train_loader = DataLoader(MNIST(mode='train'), **params.dataloader, collate_fn=AutoCollate(self.device))
            test_loader = DataLoader(MNIST(mode='test'), **params.dataloader, collate_fn=AutoCollate(self.device))
        elif params.dataset == 'fmnist':
            train_loader = DataLoader(FMNIST(mode='train'), **params.dataloader,
                                      collate_fn=AutoCollate(self.device))
            test_loader = DataLoader(FMNIST(mode='test'), **params.dataloader,
                                     collate_fn=AutoCollate(self.device))
        else:
            assert False

        self.regist_databundler(
            train=train_loader,
            test=test_loader,
        )

    def models(self, params: Params):
        from torch.optim import SGD
        from arch import cnn13, simple

        if params.arch == 'cnn13':
            self.model = cnn13.CNN13()
        elif params.arch == 'simple':
            self.model = simple.SimpleNet()
        else:
            assert False

        if params.ema:
            from tricks.ema import ema
            self.ema_model = ema(self.model)

        self.to(self.device)
        self.optim = SGD(self.model.parameters(), **params.optim)

    def train_batch(self, eidx, idx, global_step, batch_data, params, device):
        optim = self.optim
        meter = Meter()
        xs, ys = batch_data

        # 训练逻辑
        logits = self.model(xs)
        meter.loss = F.cross_entropy(logits, ys)
        self.train_precise(logits, ys, meter)
        # 反向传播
        meter.loss.backward()
        optim.step()
        optim.zero_grad()

        return meter

    def on_train_batch_end(self, trainer: 'BaseTrainer', func, params: Params, meter: Meter, *args, **kwargs):
        if params.ema:
            from tricks.ema import update_ema_variables
            update_ema_variables(self.model, self.ema_model, params.global_step)

    def train_precise(self, logits, labels, meter):
        """train batch accuracy"""
        with torch.no_grad():
            _, maxid = torch.max(logits, dim=-1)
            total = labels.size(0)
            top1 = (labels == maxid).sum().item()
            meter.acc = top1 / total
            meter.percent(meter.acc_)
        return meter.acc

    def test_eval_logic(self, dataloader, param: Params):
        from thexp.calculate import accuracy as acc
        with torch.no_grad():
            count_dict = Meter()
            for xs, labels in dataloader:
                xs, labels = xs.to(self.device), labels.to(self.device)
                preds = self.predict(xs)
                total, topk_res = acc.classify(preds, labels, topk=param.topk)
                count_dict["total"] += total
                for i, topi_res in zip(param.topk, topk_res):
                    count_dict[i] += topi_res
        return count_dict

    def predict(self, xs):
        if params.ema:
            return self.ema_model(xs)
        else:
            return self.model(xs)