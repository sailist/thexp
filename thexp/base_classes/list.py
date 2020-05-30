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
from collections.abc import Iterable

class llist(list):
    """
    添加了根据 可迭代对象切片的功能
    Examples
    >>> res = llist([1,2,3,4])
    ... print(res[0,2,1])
    """
    def __getitem__(self, i: [int,slice,Iterable]):
        if isinstance(i,(slice,int)):
            return super().__getitem__(i)
        elif isinstance(i,Iterable):
            return [self.__getitem__(id) for id in i]

