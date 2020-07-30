"""

"""
import sys

sys.path.insert(0, r"E:\Python\iLearn\thexp")
from thexp import __VERSION__

print(__VERSION__)

# glob.add_value('datasets', '/home/share/yanghaozhe/pytorchdataset', glob.LEVEL.repository)

from config import params
from trainer.mnisttrainer import MNISTTrainer

# params
for p in params.grid_search("optim.lr", [0.001, 0.005, 0.0005]):
    for pp in p.grid_search("epoch", [10, 15, 20]):
        trainer = MNISTTrainer(pp)
        trainer.train()
