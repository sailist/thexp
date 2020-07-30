import importlib

from torch.optim.optimizer import Optimizer

from .attr import attr


class OptimParams(attr):
    def __init__(self, name: str = None, **kwargs):
        super().__init__()
        self.args = attr.from_dict(kwargs)  # type:attr
        self.name = name

    def build(self, parameters, optim_cls=None) -> Optimizer:
        lname = self.name.lower()
        if optim_cls is None:
            assert self.name is not None
            optim_lib = importlib.import_module("torch.optim.{}".format(lname))
            optim_cls = getattr(optim_lib, self.name, None)

        return optim_cls(parameters, **self.args)


class ScheduleParams(attr):
    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.name = name
        self.args = attr.from_dict(kwargs)

    def build(self):
        pass



class ParamsFactory:
    @staticmethod
    def _filter_none(**kwargs):
        return {k: v for k, v in kwargs.items() if v is not None}

    @staticmethod
    def create_optim(name, **kwargs):
        kwargs = ParamsFactory._filter_none(**kwargs)
        return OptimParams(name, **kwargs)

    @staticmethod
    def create_scheule(name, **kwargs):
        kwargs = ParamsFactory._filter_none(**kwargs)
        return

    # opf = OptimFactory.create("SGD", lr=0.1)
