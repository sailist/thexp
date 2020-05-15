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
import random

import torch
from torch.nn import functional as F


class RandomHorizontalFlip():
    """Flip randomly the image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x: torch.Tensor):
        """NCWH"""
        if random.random() < self.p:
            x = x.flip(dims=(-1,))
        return x.contiguous()


class RandomBatchCrop():
    def __init__(self, size, padding=4, fill=0, p=0.5):
        self.p = p
        self.size = size
        self.pad = Pad(padding, fill=fill)

    def __call__(self, x: torch.Tensor):
        """NCWH"""
        x = self.pad(x)

        h, w = x.shape[-2:]
        new_h = new_w = self.size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        x = x[:, :, top: top + new_h, left: left + new_w]
        return x


class Pad():
    def __init__(self, padding=4, fill=0):
        self.pad = padding
        self.fill = fill

    def __call__(self, x: torch.Tensor):
        x = F.pad(x, [self.pad] * 4)
        return x
