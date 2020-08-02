"""
Templete
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thexp import Trainer, Meter
from torch.nn import functional as F

from trainers import GlobalParams
from trainers.mixin import *


class MyTrainer(callbacks.BaseCBMixin,
                datasets.BaseDatasetMixin,
                models.ModelMixin,
                acc.ClassifyAccMixin,
                losses.Loss,
                tricks.TrickMixin,
                Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: GlobalParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        xs, ys = batch_data

        logits = self.to_logits(xs)

        meter.Lce = F.cross_entropy(logits, ys)

        self.any_()

        # self.optim.zero_grad()
        # meter.Lce.backward()
        # self.optim.step()

        self.acc_precise_(logits, ys, meter, name='acc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        raise NotImplementedError()


if __name__ == '__main__':
    params = GlobalParams()
    # params.device = 'cuda:0'
    params.from_args()

    trainer = MyTrainer(params)
    trainer.train()
    trainer.save_model()
