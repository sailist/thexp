from torch import nn
import torch
from copy import deepcopy
from torch import FloatTensor, LongTensor
from torch.cuda import LongTensor as CLongTensor


class EMA():
    def __init__(self, module: nn.Module, alpha=0.999):
        self.ema = deepcopy(module)
        [i.detach_() for i in self.ema.parameters()]
        self.module = module
        self.alpha = 0.999

    def step(self, alpha=None):
        ema_model, model = self.ema, self.module
        if alpha is None:
            alpha = self.alpha

        with torch.no_grad():
            for (_, ema_param), (_, param) in zip(ema_model.state_dict().items(), model.state_dict().items()):
                if not isinstance(param, (LongTensor, CLongTensor)):
                    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
                else:
                    ema_param.data.copy_(alpha * ema_param + (1 - alpha) * param)

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.ema(*args, **kwargs)
