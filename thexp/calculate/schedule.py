"""
This module aims to provide an easy and general method to generate series values, which can be used in
 applying learning rate of optimizer or scaling a loss value as weight.

To meet this demand, the concept 'schedule' is summarized as a periodic math function, which have
left/right interval and  start/end value.

Hence, a `Schedule` class is provided, it receives four base parameters: start, end, left, and right corresponding to
the above concept respectively.

This class provides some common methods. For example, when you have a schedule instance, you can apply learning rate by
simple call `schedule.scale()` or `schedule.apply()` function.

And, you can use `schedule.plot()` to plot a curve of the values in each step. The plot function use `matplotlib` as backend.

If you don't want to create an instance, you can call classmethod `get(cls, cur, start=0, end=1, left=0, right=1)` to get
value.

Except for the base `Schedule` class, some other subclasses of `Schedule`  which may be general used is provided, too. All can
 be easily understand by their names and plot curves.
"""
import numpy as np
from thexp.base_classes.attr import attr


class Schedule(attr):
    """
    ratio 变化为从 1 - 0
    """

    # __schdule_name__ = None

    def __init__(self, start=0, end=1, left=0, right=1, *args, **kwargs):
        super().__init__()
        self.right = right
        self.left = left
        self.start = start
        self.end = end

    def ratio(self, cur):
        return (cur - self.left) / (self.right - self.left)

    def __call__(self, cur):
        raise NotImplementedError()

    def plot(self, num=1000, left=None, right=None):
        """
        Plot a curve of the schedule.
        Args:
            num:
            left:
            right:

        Returns:
            plt.plot

        Notes:
            You may need to call `plt.show`() to show it.
        """
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
        Scale the learning rate by current value. 'scale' means not apply the current schedule value directly to
        the learning rate, but multiple the initial learning rate. You can use `schedule.apply()` to apply the schedule
        value directly.

        Notes:
        -------
        When first apply scale function, a `_raw_lr` represent initial lr will be set in each param_group, then, the
        learning rate(store in param_groups with key 'lr') will be calculated by `_raw_lr * schedule(cur)`.

        Args:
            optimizer: A pytorch optimizer instance.
            cur: current step of this schedule.

        Returns:
            Current schedule value.
        """
        ratio = self(cur)
        for param_group in optimizer.param_groups:  # type:dict
            raw_lr = param_group.setdefault('_raw_lr', param_group['lr'])
            param_group['lr'] = raw_lr * ratio

        return ratio

    def apply(self, optimizer, cur):
        """
        Apply the learning rate with current schedule value.

        Args:
            optimizer: A pytorch optimizer instance.
            cur: current step of this schedule.

        Returns:

        """
        new_lr = self(cur)
        for param_group in optimizer.param_groups:  # type:dict
            param_group['lr'] = new_lr

        return new_lr

    @classmethod
    def get(cls, cur, start=0, end=1, left=0, right=1, *args, **kwargs):
        """get the current schedule value without create `schedule` instance. """
        return cls(start=0, end=1, left=0, right=1, *args, **kwargs)(cur)


class CosSchedule(Schedule):
    """one cycle cosine functoin"""

    def __call__(self, cur):
        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        ratio = self.ratio(cur)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi))
        return self.start * cos_ratio + self.end * (1 - cos_ratio)


class PeriodCosSchedule(Schedule):
    """
    periodic cosine schedule
    """

    def __call__(self, cur):
        cur = float(cur - self.left) % (self.right - self.left)
        ratio = self.ratio(cur + self.left)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi * 2))
        return self.start * cos_ratio + self.end * (1 - cos_ratio)


class HalfPeriodCosSchedule(Schedule):
    """
    half periodic cosine schedule
    """

    def __call__(self, cur):
        cur = float(cur - self.left) % (self.right - self.left)
        ratio = self.ratio(cur + self.left)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi))
        return self.start * cos_ratio + self.end * (1 - cos_ratio)


class LinearSchedule(Schedule):
    """linear schedule"""

    def __call__(self, cur):
        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        linear_ratio = self.ratio(cur)
        return self.start * (1 - linear_ratio) + self.end * linear_ratio
