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

from typing import Any

class Merge(type):
    """
    元类，用于将子类和父类共有字典，集合时，子类的覆盖行为改为合并父类的字典，集合

    由于用途特殊，仅识别类变量中以下划线开头的变量
    ::
        class A(metaclass=Merge):
            _dicts = {"1": 2, "3": 4}

        class B(A):
            _dicts = {"5":6,7:8}

        print(B._dicts)

    result:
    >>> {'5': 6, '3': 4, '1': 2, 7: 8}
    """

    def __new__(cls, name, bases, attrs: dict, **kwds):
        for base in bases:
            for key, value in base.__dict__.items():  # type:(str,Any)
                if not key.startswith("_"):
                    continue
                if isinstance(value, set):
                    v = attrs.setdefault(key, set())
                    v.update(value)
                elif isinstance(value, dict):
                    v = attrs.setdefault(key, dict())
                    v.update(value)

        return type.__new__(cls, name, bases, dict(attrs))

