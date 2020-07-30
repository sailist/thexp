"""

"""

import sys
sys.path.insert(0,"../../")
from thexp.utils.screen import ScreenStr
import time

s = "\rLong Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text;Long Text"
for i in range(100):
    print(ScreenStr(s,leftoffset=10),end="")
    time.sleep(0.2)


from thexp.contrib.data.collate import AutoCollate
from torch.utils.data.dataloader import DataLoader
import torch
device = torch.device('cuda:1')
DataLoader(...,collate_fn=AutoCollate(device))
