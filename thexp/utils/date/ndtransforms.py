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
import torch

# from torchvision.transforms import *


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class RandomVerticalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = np.flip(x, axis=0)

        return x


class ToNumpy():
    def __call__(self, x):
        return np.array(x)


class ToHWC():
    def __call__(self, x:np.ndarray):
        if x.ndim == 2:
            return x
        if x.shape[-1] <=4 :
            return x
        return x.transpose([2,0,1])

class ToCHW():
    def __call__(self, x:np.ndarray):
        if x.ndim == 2:
            return x
        if x.shape[-1] >4 :
            return x
        return x.transpose([2,0,1])

class RandomHorizontalFlip():
    """Flip randomly the image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = np.fliplr(x)
        return x.copy() # directly return x will cause  'negative' error when doing Torch.from_numpy

class RandomCrop():
    def __init__(self, size, padding=4, fill=0):
        self.size = size
        self.pad = Pad(pad=padding, fill=fill)

    def __call__(self, x):
        x = self.pad(x)

        if x.ndim == 2:
            h, w = x.shape
        elif x.shape[-1] <= 4:
            h, w = x.shape[:-1]
        else:
            h, w = x.shape[1:]

        new_h = new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        if x.ndim == 2:
            x = x[top: top + new_h, left: left + new_w]
        elif x.shape[-1] <= 4:
            x = x[top: top + new_h, left: left + new_w, :]
        else:
            x = x[:, top: top + new_h, left: left + new_w]

        return x


class Pad():
    def __init__(self, pad=4, fill=0):
        self.pad = pad
        self.fill = fill

    def __call__(self, x):
        if x.ndim == 2:
            x = np.pad(x, self.pad, mode="constant")
        else:
            if x.shape[-1] <= 4:
                x = np.pad(x, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="constant",
                           constant_values=self.fill)
            else:
                x = np.pad(x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="constant",
                           constant_values=self.fill)
        return x
