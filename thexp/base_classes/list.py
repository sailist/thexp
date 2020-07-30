"""

"""
from collections.abc import Iterable
import torch
import numbers


class llist(list):
    """
    添加了根据 可迭代对象切片的功能
    Examples:
    >>> res = llist([1,2,3,4])
    ... print(res[0])
    ... print(res[0:3])
    ... print(res[0,2,1])

    >>> idx = torch.randperm(3)
    ... print(res[idx])

    >>> idx = torch.randint(0,3,[3,4])
    ... print(res[idx])
    """

    def __getitem__(self, i: [int, slice, Iterable]):
        if isinstance(i, (slice, numbers.Integral)):
            # numpy.int64 is not an instance of built-in type int
            res = super().__getitem__(i)
            if isinstance(res, list):
                return llist(res)
            else:
                return res
        elif isinstance(i, Iterable):
            if isinstance(i, torch.Tensor):
                if len(i.shape) == 0:
                    i = i.item()
                    return self.__getitem__(i)
                else:
                    i = i.tolist()
            return llist(self.__getitem__(id) for id in i)
