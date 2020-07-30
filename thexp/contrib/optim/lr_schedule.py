"""

"""
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

"""deprecated"""
def rampup_cos_schedule(optimizer, epoch, min_lr, max_lr, rampup=10, return_func=False):
    """
    rampup_cos_schedule(optimizer,400,0.0001,0.1, rampup=5,return_func=False)
    :param optimizer:
    :param epoch:
    :param min_lr:
    :param max_lr:
    :param rampup:
    :param return_func:
    :return:
    """

    def wrap(cur):
        if cur < rampup:
            return min_lr + (max_lr - min_lr) * (cur / rampup)
        else:
            return 0.5 * (1 + np.cos(cur * np.pi / (epoch - rampup))) * max_lr

    if return_func:
        return wrap

    return LambdaLR(optimizer, lr_lambda=wrap)
