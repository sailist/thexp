from torch import nn
import torch
from copy import deepcopy
from torch import FloatTensor, LongTensor
from torch.cuda import LongTensor as CLongTensor


def EMA(model: nn.Module, alpha_=0.999):
    """
    Exponential Moving Average for nn.Module
    Args:
        model:
        alpha_:

    Returns:

    """
    ema_model = deepcopy(model)
    [i.detach_() for i in ema_model.parameters()]

    def step(alpha=None):

        if alpha is None:
            alpha = alpha_

        with torch.no_grad():
            for (_, ema_param), (_, param) in zip(ema_model.state_dict().items(), model.state_dict().items()):
                ema_param.to(param.device)
                if not isinstance(param, (LongTensor, CLongTensor)):
                    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
                else:
                    ema_param.data.copy_(alpha * ema_param + (1 - alpha) * param)

    forward_ = ema_model.forward

    def forward(*args, **kwargs):
        with torch.no_grad():
            return forward_(*args, **kwargs)

    ema_model.forward = forward
    ema_model.step = step
    return ema_model