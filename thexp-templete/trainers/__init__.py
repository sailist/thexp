from thexp import Params


class GlobalParams(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 400
        self.optim = self.create_optim('SGD',
                                       lr=0.05,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
        self.dataset = self.choice('dataset', 'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn')
        self.n_classes = 10
        self.topk = [1, 2, 3, 4]

        self.batch_size = 64
        self.num_workers = 4

        self.ema = True
        self.ema_alpha = 0.999


