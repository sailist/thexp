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
import sys

sys.path.insert(0, r"E:\Python\iLearn\thexp")
from thexp import __VERSION__

print(__VERSION__)
from torch.nn import functional as F
from thexp import *
import torch.nn as nn
import torch

# glob.add_value('datasets', '/home/share/yanghaozhe/pytorchdataset', glob.LEVEL.repository)

params = Params()
params.device = 'cuda:1'
params.epoch = 5
params.batch_size = 128
params.topk = (1, 4)
params.from_args()
# params.root = glob['datasets']
params.root = '/home/share/yanghaozhe/pytorchdataset'
params.dataloader = dict(shuffle=True, batch_size=32, drop_last=True)
params.optim = dict(lr=0.01, weight_decay=0.09, momentum=0.9)


def MNIST(mode="train", download=False):
    # 由于不同数据集区分训练或测试集的方式不同，因此建议数据集以统一的接口定义
    from torchvision.datasets.mnist import MNIST
    from torchvision import transforms
    weak = transforms.ToTensor()
    if mode == "train":
        return MNIST(root=params.root, train=True, download=download, transform=weak)
    else:
        return MNIST(root=params.root, train=False, transform=weak, download=download)


class SimpleNet(nn.Module):
    """used in MNIST"""

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.Tanh(),
                                nn.Linear(256, 128),
                                nn.Tanh(),
                                nn.Linear(128, 10)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class MNISTTrainer(Trainer):
    __exp_name__ = "MNISTDemo"

    def callbacks(self, params: Params):
        callbacks.LoggerCallback().hook(self)
        callbacks.AutoRecord().hook(self)
        callbacks.EvalCallback(test_per_epoch=5).hook(self)

    def datasets(self, params: Params):
        from torch.utils.data.dataloader import DataLoader
        from thexp.torch.data.collate import AutoCollate

        # self.device = torch.device(param.device)
        train_loader = DataLoader(MNIST(mode='train'), **params.dataloader, collate_fn=AutoCollate(self.device))
        test_loader = DataLoader(MNIST(mode='test'), **params.dataloader, collate_fn=AutoCollate(self.device))

        self.regist_databundler(
            train=train_loader,
            test=test_loader,
        )

    def models(self, params: Params):
        from torch.optim import SGD
        self.model = SimpleNet().to(self.device)
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
        return self.model(xs)


# params
for p in params.grid_search("optim.lr", [0.001, 0.005, 0.0005]):
    for pp in p.grid_search("epoch", [10, 15, 20]):
        trainer = MNISTTrainer(pp)
        trainer.train()
