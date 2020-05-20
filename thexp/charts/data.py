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

class Chart:
    def __init__(self):
        self.options = {}

    def _check_name(self, name):
        i = 0
        while name in self.options:
            name = '{}-{}'.format(name,i)
        return name
class Curve(Chart):

    def add_series(self,values,steps,name=None):
        name = self._check_name(name)
        self.options[name] = {
            'y':values,
            'x':steps,
        }

class Bar(Chart):
    pass

class Parallel:
    pass

