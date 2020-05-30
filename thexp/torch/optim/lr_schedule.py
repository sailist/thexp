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
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


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
