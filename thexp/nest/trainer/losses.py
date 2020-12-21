"""
for calculating and recording loss.
"""
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


# class TripletLoss(Loss):
# def loss_l2_reg_(self, tensors: List[torch.Tensor], w_l2=1,
#                  meter: Meter = None, name: str = 'Lreg'):
#     loss = sum([(tensor ** 2).sum(dim=-1).mean() for tensor in tensors]) * w_l2
#     if meter is not None:
#         meter[name] = loss
#     return loss


class SimCLRLoss(Loss):
    def loss_sim_(self, features: torch.Tensor,
                 temperature=0.5,
                 meter: Meter = None,
                 name: str = 'Lsim'):
        """

        :param features: [batchsize, 2, feature_dim]
        :param temperature:
        :param meter:
        :param name:
        :return:
        """
        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / temperature # # 两两之间的相似度

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach() # 两两相似度 - 第一次增广的自身的相似度

        mask = mask.repeat(1, 2)

        # 排除第一次增广自身相似度的 mask，# 也即分母部份 logits 的 mask
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask # 第一次样本增广和第二次样本增广的相似度的 mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask # 分母部份的底

        # logits 的 exp 和 log 抵消了，所以只要logits本身即可，而 第二项则是 log 内的分母
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        # mask * log_prob 的 每行都表示 分子为正例，分母为全部的一个 lij，最后求和得到所有 lij 的损失
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        if meter is not None:
            meter[name] = loss

        return loss
