"""

"""
from thexp import Params

params = Params()
params.device = 'cuda:1'
params.epoch = 5
params.batch_size = 128
params.topk = (1, 4)
params.from_args()
params.root = '/home/share/yanghaozhe/pytorchdataset'
params.dataloader = dict(shuffle=True, batch_size=32, drop_last=True)
params.optim = dict(lr=0.01, weight_decay=0.09, momentum=0.9)
params.choice('dataset', 'mnist', 'fmnist')
params.dataset = 'mnist'
params.bind('dataset', 'mnist', 'arch', 'simple')
params.bind('dataset', 'fmnist', 'arch', 'simple')
params.bind('dataset', 'cifar10', 'arch', 'cnn13')
params.ema = True
