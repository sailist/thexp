from thexp import Trainer
from thexp.frame import callbacks

from .. import GlobalParams


class __CB(callbacks.TrainCallback):
    pass


class CBMixin(Trainer):
    def callbacks(self, params: GlobalParams):
        from thexp import callbacks
        callbacks.LoggerCallback().hook(self)
        callbacks.EvalCallback(1, 1).hook(self)
        callbacks.AutoRecord().hook(self)
