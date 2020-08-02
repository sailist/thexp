"""

"""

from thexp.base_classes.attr import _attr_clss, attr
from thexp import Params
from thexp.calculate.schedule import LinearSchedule

p = Params()
p.optim = p.create_optim('SGD', lr=0.1)
p.sch = LinearSchedule(right=10)
p.to_json('pj')

print(p)

np = Params()
np.from_json('pj')

print(np)
# a = attr()
#
#
#
# print(a)
#
# print(_attr_clss)
