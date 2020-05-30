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
import copy
from collections import OrderedDict
from typing import Any


class attr(OrderedDict):
    def __getattr__(self, item):
        if item not in self or self[item] is None:
            self[item] = attr()
        return self[item]

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = attr.from_dict(value)
        self[name] = value

    def __getitem__(self, k):
        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            return super().__getitem__(ks[0])

        cur = self
        for tk in ks:
            cur = cur.__getitem__(tk)
        return cur

    def __setitem__(self, k, v):
        if isinstance(v, dict):
            v = attr.from_dict(v)

        k = str(k)
        ks = k.split(".")
        if len(ks) == 1:
            super().__setitem__(ks[0], v)
        else:
            cur = self
            for tk in ks[:-1]:
                cur = cur.__getattr__(tk)
            cur[ks[-1]] = v

    def __contains__(self, o: object) -> bool:
        try:
            _ = self[o]
            return True
        except:
            return False

    @staticmethod
    def from_dict(dic: dict):
        res = attr()
        for k, v in dic.items():
            if isinstance(v, dict):
                v = attr.from_dict(v)
            res[k] = v
        return res

    def walk(self):
        for k,v in self.items():
            if isinstance(v,attr):
                for ik,iv in v.walk():
                    ok = "{}.{}".format(k,ik)
                    yield ok,iv
            else:
                yield k,v

    def __copy__(self):
        return attr(
            **{k: copy.copy(v) for k, v in self.items()}
        )

    def hash(self) -> str:
        from ..utils.generel_util import hash
        return hash(self)

    def jsonify(self) ->dict:
        """
        获取可被json化的dict，目前仅支持
        :return:
        """
        import numbers
        res = dict()
        for k,v in self.items():
            if isinstance(v,(numbers.Number,str)):
                res[k] = v
            elif isinstance(v,(set,list)):
                v = [vv for vv in v if isinstance(vv,(numbers.Number,str))]
                res[k] = v
            elif isinstance(v,attr):
                v = v.jsonify()
                res[k] = v

        return res