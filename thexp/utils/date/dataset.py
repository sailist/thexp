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
import torch
from PIL import Image
from torch import randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset


def sequence_split(dataset, lengths):
    indices = torch.arange(0, sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(_accumulate(lengths), lengths)]


def random_split_with_seed(dataset, lengths, seed=0):
    from torch.utils.data.dataset import random_split
    from .. import random
    random.fix_seed(seed)

    return random_split(dataset, lengths)


def split_with_indices(dataset, lengths, indices):
    r"""
    Randomly split a dataset into non-overlapping new nddatasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    return [Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(_accumulate(lengths), lengths)], indices


class SemiDataset(Dataset):
    def __init__(self, datas, targets, index, transform=None, target_transform=None):
        self.datas = datas[index]
        self.targets = targets[index]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img, target = self.datas[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.datas)

    @staticmethod
    def create_indice(self, *lengths, random=True):
        if random:
            indices = randperm(sum(lengths))
        else:
            indices = torch.arange(0, sum(lengths)).tolist()
        return indices


class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.trasform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):

        img, target = self.dataset[self.indices[idx]]
        if self.trasform is not None:
            img = self.trasform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)


def ratio2length(total_len, *ratios):
    assert sum(ratios) == 1
    return [total_len * i for i in ratios]


class HookMixin():
    def on(self,*args,**kwargs):
        pass



