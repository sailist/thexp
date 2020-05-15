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

from thexp.base_classes.metaclasses import Merge

def test_merge():
    class A(metaclass=Merge):
        _item = {1:2,3:4}

    class B(A):
        _item = {5:6,7:8}

    b = B()
    assert 1 in b._item and 3 in b._item and 5 in b._item and 7 in b._item
    assert 1 in B._item and 3 in B._item and 5 in B._item and 7 in B._item

