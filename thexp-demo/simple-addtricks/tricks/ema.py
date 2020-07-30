"""

"""

from copy import deepcopy

import torch
import torch.nn as nn
from torch import LongTensor
from torch.cuda import LongTensor as CLongTensor


def ema(model: nn.Module):
    ema = deepcopy(model)
    for param in ema.parameters():
        param.detach_()
    return ema


def update_ema_variables(model, ema_model, global_step, alpha=0.99):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        for (_, ema_param), (_, param) in zip(ema_model.state_dict().items(), model.state_dict().items()):
            if global_step == 1:
                ema_param.data.copy_(alpha * ema_param + (1 - alpha) * param)
            else:
                if not isinstance(param, (LongTensor, CLongTensor)):
                    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
