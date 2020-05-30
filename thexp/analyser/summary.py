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

class ValueSummary():
    def __init__(self,name,values):
        self.name = name
        self.values = np.array(values)

    @property
    def min(self):
        return self.values.min()

    @property
    def max(self):
        return self.values.max()

    @property
    def mean(self):
        return self.values.mean()

    @property
    def std(self):
        return self.values.std()

    def to_attr(self):
        from ..base_classes.attr import attr
        res = attr()
        res['{}-MIN'.format(self.name)] = self.min
        res['{}-MAX'.format(self.name)] = self.max
        res['{}-MEAN'.format(self.name)] = self.mean
        res['{}-STD'.format(self.name)] = self.std
        return res