from typing import List

from torch.nn import functional as F

import torch
from thexp import Meter


class Loss():
    def to_mid(self, xs) -> torch.Tensor:
        """input2mid"""
        raise NotImplementedError()

    def to_logits(self, xs) -> torch.Tensor:
        """input2logits"""
        raise NotImplementedError()

    def mid_to_logits(self, xs) -> torch.Tensor:
        raise NotImplementedError()


class MSELoss(Loss):
    def loss_mse_(self, logits: torch.Tensor, targets: torch.Tensor,
                  w_mse=1,
                  meter: Meter = None, name: str = 'Lmse'):
        loss = torch.mean((F.softmax(logits, dim=1) - targets) ** 2) * w_mse
        if meter is not None:
            meter[name] = loss
        return loss

    def loss_mse_with_labels_(self, logits: torch.Tensor, labels: torch.Tensor,
                              meter: Meter, name: str = 'Lmse', w_mse=1):
        from thexp.calculate.tensor import onehot
        targets = onehot(labels, logits.shape[-1])
        meter[name] = torch.mean((F.softmax(logits, dim=1) - targets) ** 2) * w_mse
        return meter[name]


class CELoss(Loss):
    def loss_ce_(self, logits: torch.Tensor, labels: torch.Tensor,
                 w_ce=1,
                 meter: Meter = None, name: str = 'Lce'):
        loss = F.cross_entropy(logits, labels) * w_ce
        if meter is not None:
            meter[name] = loss
        return loss

    def loss_ce_with_targets_(self,
                              logits: torch.Tensor, targets: torch.Tensor,
                              w_ce=1,
                              meter: Meter = None, name: str = 'Lce'):
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1)) * w_ce
        if meter is not None:
            meter[name] = loss
        return loss


class MinENTLoss(Loss):
    def loss_minent_(self, logits, w_ent=1,
                     meter: Meter = None, name: str = 'Lent'):
        loss = - torch.sum(F.log_softmax(logits, dim=-1) * F.softmax(logits, dim=-1), dim=-1).mean() * w_ent
        if meter is not None:
            meter[name] = loss
        return loss


class L2Loss(Loss):
    def loss_l2_reg_(self, tensors: List[torch.Tensor], w_l2=1,
                     meter: Meter = None, name: str = 'Lreg'):
        loss = sum([(tensor ** 2).sum(dim=-1).mean() for tensor in tensors]) * w_l2
        if meter is not None:
            meter[name] = loss
        return loss
