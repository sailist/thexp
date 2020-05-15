"""
_licence_content_
"""
import torch
import torch.nn as nn
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
import arch
from thexp import Meter, Params, Trainer,DataBundler
from thexp.utils.torch import optim
from torchvision import transforms

class _trainer_name_Param(Params):
    def __init__(self):
        super().__init__()
        self.batch_size = 100
        self.test_in_per_epoch = 10
        self.epoch = 500
        self.arch = "cnn13"

        self.mixup = True
        self.mixup_unsup = False
        self.mix_unsup_weight = True
        self.mixup_consistency = 100
        self.ent_consistency = 10
        self.adjust_optimizer_lr = True
        self.eval_ratio = 0.1


class _trainer_name_Meter(Meter):
    def __init__(meter):
        super().__init__()


class _trainer_name_Trainer(Trainer):
    def initial_trainer(self, params: Params):
        self.model = arch.MLP(10)
        self.optim = optim.SGD(params=self.model.parameters(),
                               lr=0.1,
                               momentum=0.9,
                               weight_decay=0.0001,
                               nesterov=False)

        dataset = FakeData(image_size=(28, 28), transform=transforms.ToTensor())
        train_loader = eval_loader = test_loader = DataLoader(dataset, shuffle=True, batch_size=32, drop_last=True)

        self.regist_databundler(
            train=train_loader,
            test=test_loader,
            eval=eval_loader,
        )
        self.to(self.device)
        self.lossf = nn.CrossEntropyLoss()

    def train_batch(self, eidx, idx, global_step, batch_data, params: _trainer_name_Param, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)

        meter = Meter()
        model = self.model
        optim = self.optim

        xs, ys = batch_data
        xs, ys = xs.to(device), ys.to(device)

        logits = model(xs)

        meter.loss = self.lossf(logits, ys)

        optim.zero_grad()
        meter.loss.backward()
        optim.step()

        return meter

    def predict(self, xs):  # 用于测试和验证
        return self.model(xs)
