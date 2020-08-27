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

from typing import List


class Schedule(attr):
    """
    ratio 变化为从 1 - 0
    """

    # __schdule_name__ = None

    def __init__(self, start=0., end=1., left=0, right=1, *args, **kwargs):
        super().__init__()
        self.left = left
        self.right = right
        self.start = start
        self.end = end
        self.constant = False

    def toggle_constant(self, toggle=True):
        """fix the schedule as the first value"""
        self.constant = toggle
        return self

    def ratio(self, cur):
        if self.constant:
            return 0
        return (cur - self.left) / (self.right - self.left)

    def __call__(self, cur):
        raise NotImplementedError()

    def plot(self, num=1000, left=None, right=None, show=True):
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

        res = plt.plot(x, y)
        if show:
            plt.show()
        return res

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
    def get_val(cls, cur, start=0, end=1, left=0, right=1, *args, **kwargs):
        """get the current schedule value without create `schedule` instance. """
        return cls(start=0, end=1, left=0, right=1, *args, **kwargs)(cur)


class CosSchedule(Schedule):
    """one cycle cosine functoin"""

    def __call__(self, cur):
        if self.constant:
            return self.start

        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        ratio = self.ratio(cur)
        cos_ratio = 0.5 * (1 + np.cos(ratio * np.pi))
        return self.start * cos_ratio + self.end * (1 - cos_ratio)


class ConstantSchedule(Schedule):
    def __init__(self, value=0.5, *args, **kwargs):
        super().__init__(start=value, end=value, *args, **kwargs)

        self.constant = True


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
        if self.constant:
            return self.start

        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        linear_ratio = self.ratio(cur)
        return self.start * (1 - linear_ratio) + self.end * linear_ratio


class ExpSchedule(Schedule):
    """slow to quick"""

    def __call__(self, cur):
        if self.constant:
            return self.start

        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        ratio = self.ratio(cur)
        residual = np.exp(-5)

        exp_ratio = np.exp((ratio - 1) * 5) - residual * (1 - ratio)
        return self.start * (1 - exp_ratio) + self.end * exp_ratio


class LogSchedule(Schedule):
    """quick to slow"""

    def __call__(self, cur):
        if self.constant:
            return self.start

        if cur < self.left:
            return self.start
        elif cur > self.right:
            return self.end

        ratio = self.ratio(cur)

        residual = np.exp(-5)

        log_ratio = 1 - np.exp(-ratio * 5) + residual * ratio
        return self.start * (1 - log_ratio) + self.end * log_ratio


class ScheduleList(attr):
    def __init__(self, schedules=None, bound='left'):
        super().__init__()
        if schedules is None:
            schedules = []
        self.bound = bound
        # assert len(schedules) > 0
        if bound == 'left':
            self.schedules = sorted(schedules, key=lambda x: x.left)
        elif bound == 'right':
            self.schedules = sorted(schedules, key=lambda x: x.right)
        else:
            assert False
        self.left = self.schedules[0].left
        self.right = self.schedules[-1].right
        # super().__init__(None, None, left, right, *args, **kwargs)

    def __call__(self, cur):
        if self.bound == 'left':
            for i, schedule in enumerate(self.schedules):
                if i + 1 < len(self.schedules):
                    if cur < self.schedules[i + 1].left:
                        return schedule(cur)
                    else:
                        continue
                else:
                    return schedule(cur)
        elif self.bound == 'right':
            for i, schedule in enumerate(self.schedules):
                if cur < self.schedules[i].right:
                    return schedule(cur)
            return schedule(cur)
        else:
            assert False
