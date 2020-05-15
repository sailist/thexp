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

from thexp.utils.random import *

def test_state():
    state_dict = fix_seed(10)
    a,b,c = random.randint(0,10),np.random.rand(1),torch.rand(1)
    set_state(state_dict)
    d,e,f = random.randint(0,10),np.random.rand(1),torch.rand(1)

    assert a==d and b==e and c==f

