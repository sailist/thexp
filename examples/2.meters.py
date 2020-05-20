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
import sys
sys.path.insert(0,"../")
from thexp import __VERSION__
print(__VERSION__)


from thexp import Meter,AvgMeter
import torch

m = Meter()
m.a = 1
m.b = "2"
m.c = torch.rand(1)[0]

m.c1 = torch.rand(1)
m.c2 = torch.rand(2)
m.c3 = torch.rand(4, 4)
print(m)

m = Meter()
m.a = 0.236
m.b = 3.236
m.c = 0.23612312
m.percent(m.a_)
m.int(m.b_)
m.float(m.c_,2)
print(m)


am = AvgMeter()
for j in range(5):
    for i in range(100):
        m = Meter()
        m.percent(m.c_)
        m.a = 1
        m.b = "2"
        m.c = torch.rand(1)[0]

        m.c1 = torch.rand(1)
        m.c2 = torch.rand(2)
        m.c3 = torch.rand(4, 4)
        m.d = [4]
        m.e = {5: "6"}
        # print(m)
        am.update(m)
    print(am)

from thexp import Meter
m = Meter()
print(m.k)

m.all_loss = m.all_loss + 5
m.all_loss = m.all_loss + 3
print(m.all_loss)