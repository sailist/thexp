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
import operator as op
import sys
from functools import reduce

import numpy as np

def pin(*sizes, dtype="float32",format=False):
    """
    计算相应尺寸和类型“至少”占用的内存大小

    由于除了存放相应 size 个数的数据外，还有用于辅助的各类变量，因此说是至少
    :param sizes:
    :param dtype:
    :param format:
    :return:
    """
    # len(sizes) * 16 + 80
    one = np.array(1, dtype=dtype)

    size = sys.getsizeof(one) - 80
    mem = reduce(op.mul, sizes) * size

    if format:
        unit = "B"
        if mem > 1024:
            mem /= 1024
            unit = "KB"
            if mem > 1024:
                mem /= 1024
                unit = "MB"
                if mem > 1024:
                    mem /= 1024
                    unit = "GB"
        return "{:.4f} {}".format(mem,unit)

    return mem  # byte


if __name__ == '__main__':
    print(pin(10,50000,format=True))