import numpy as np
from thexp.base_classes.attr import attr


class Schedule(attr):
    """
    ratio 变化为从 1 - 0
    """
    __schdule_name__ = None

    def __init__(self, start=0, end=1, left=0, right=1, *args, **kwargs):
        super().__init__()
        schedule_name = self.__class__.__schdule_name__
        if schedule_name is None:
            schedule_name = self.__class__.__name__
        self.schedule = schedule_name
        self.right = right
        self.left = left
        self.start = start
        self.end = end

    def ratio(self, cur):
        return (cur - self.left) / (self.right - self.left)

    def func(self, cur):
        raise NotImplementedError()

    def __call__(self, cur):
        return self.func(cur)

    def __getitem__(self, cur):
        return self.func(cur)

    def plot(self, num=1000, left=None, right=None):
        from matplotlib import pyplot as plt
        if left is None:
            left = self.left

        if right is None:
            right = self.right
        x = np.linspace(left, right, num)
        y = [self(i) for i in x]
        return plt.plot(x, y)

    def scale(self, optimizer, cur):
        """
        将当前得到的数值作为权重对 Optimzizer上的lr进行放缩

        第一次应用时，会在 param_group 新建一个 _raw_lr 用于保存初始学习率

        之后每一次scale设置均为 raw_lr * ratio

        适用于不同 param_group 学习率不同的情况
        Args:
            optimizer:
            cur:

        Returns:

        """
        ratio = self(cur)
        for param_group in optimizer.param_groups:  # type:dict
            raw_lr = param_group.setdefault('_raw_lr', param_group['lr'])
            param_group['lr'] = raw_lr * ratio

        return ratio

    def apply(self, optimizer, cur):
        """
        将当前得到的数值作为学习率应用到 Optimizer上
        Args:
            optimizer:
            cur:

        Returns:

        """
        new_lr = self(cur)
        for param_group in optimizer.param_groups:  # type:dict
            param_group['lr'] = new_lr

        return new_lr


class CosSchedule(Schedule):
    """Cos 权重"""

    def func(self, cur):
        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        ratio = self.ratio(cur)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi))
        return self.start * cos_ratio + self.end * (1 - cos_ratio)


class PeriodCosSchedule(Schedule):

    def func(self, cur):
        cur = float(cur - self.left) % (self.right - self.left)
        ratio = self.ratio(cur + self.left)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi * 2))
        return self.start * cos_ratio + self.end * (1 - cos_ratio)


class HalfPeriodCosSchedule(Schedule):

    def func(self, cur):
        cur = float(cur - self.left) % (self.right - self.left)
        ratio = self.ratio(cur + self.left)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi))
        return self.start * cos_ratio + self.end * (1 - cos_ratio)


class LinearSchedule(Schedule):
    """线性权重"""

    def func(self, cur):
        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        linear_ratio = self.ratio(cur)
        return self.start * (1 - linear_ratio) + self.end * linear_ratio
