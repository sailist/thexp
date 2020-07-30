"""

"""

from thexp.frame.meter import Meter,AvgMeter
import torch

def test_meter():
    m = Meter()
    am = AvgMeter()
    for j in range(100):
        for i in range(100):
            m = Meter()
            m.a = 1
            m.b = "2"
            m.c = torch.tensor(3)
            m.c = torch.tensor([4, 5])
            m.d = [4]
            m.e = {5: "6"}
        am.update(m)

