"""

"""
import sys

sys.path.insert(0, r"E:\Python\iLearn\thexp")
from thexp import __VERSION__

print(__VERSION__)
from torch.nn import functional as F
from thexp import *
import torch

# glob.add_value('datasets', '/home/share/yanghaozhe/pytorchdataset', glob.LEVEL.repository)

params = Params()
params.device = 'cuda:1'
params.epoch = 5
params.batch_size = 128
params.topk = (1, 4)
params.from_args()
params.root = '/home/share/yanghaozhe/pytorchdataset'
params.dataloader = dict(shuffle=True, batch_size=32, drop_last=True)
params.optim = dict(lr=0.01, weight_decay=0.09, momentum=0.9)
params.choice('dataset', 'mnist', 'fmnist')
params.dataset = 'mnist'
params.bind('dataset', 'mnist', 'arch', 'simple')
params.bind('dataset', 'fmnist', 'arch', 'simple')
params.bind('dataset', 'cifar10', 'arch', 'cnn13')
params.ema = True


class MNISTTrainer(Trainer, callbacks.TrainCallback):
    __exp_name__ = "MNISTDemo"

    def callbacks(self, params: Params):
        callbacks.LoggerCallback().hook(self)
        callbacks.AutoRecord().hook(self)
        callbacks.EvalCallback(test_per_epoch=5).hook(self)

    def datasets(self, params: Params):
        from torch.utils.data.dataloader import DataLoader
        from thexp.contrib.data.collate import AutoCollate
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


# params
for p in params.grid_search("optim.lr", [0.001, 0.005, 0.0005]):
    for pp in p.grid_search("epoch", [10, 15, 20]):
        trainer = MNISTTrainer(pp)
        trainer.train()
