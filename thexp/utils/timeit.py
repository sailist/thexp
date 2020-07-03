"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""
import pprint
import warnings
from collections import OrderedDict

import time

from thexp.utils.generel_util import curent_date


def format_second(sec: int) -> str:
    """
    将 秒（整数） 格式化为 {}h{}m{}s，其中，等于0 的部份会被隐藏
    Args:
        sec:

    Returns:

    """
    hour = min = 0
    unit = "s"

    sec, ms = divmod(sec, 1)
    if sec > 60:
        min, sec = divmod(sec, 60)
        if min > 60:
            hour, min = divmod(min, 60)
            fmt = "{}h{}m{}s".format(hour, min, int(sec))
        else:
            fmt = "{}m{}s".format(min, int(sec))
    else:
        fmt = "{}s".format(int(sec))
    return fmt


class TimeIt:
    """
    用于计算各部分用时，关键方法在 start() / mark() / end() 上

    """

    def __init__(self):
        self.last_update = None
        self.ends = False
        self.times = OrderedDict()

    def offset(self):
        now = time.time()

        if self.last_update is None:
            offset = 0
        else:
            offset = now - self.last_update

        self.last_update = now
        return offset, now

    def clear(self):
        self.last_update = None
        self.ends = False
        self.times.clear()

    def start(self):
        self.clear()
        self.mark("start", True)

    def mark(self, key, add_now=False):
        if self.ends:
            warnings.warn("called end method, please use start to restart timeit")
            return
        key = str(key)
        offset, now = self.offset()

        self.times.setdefault("use", 0)
        self.times["use"] += offset

        if add_now:
            self.times[key] = curent_date("%H:%M:%S")
        else:
            self.times.setdefault(key, 0)
            self.times[key] += offset

    def end(self):
        self.mark("end", True)
        self.ends = True

    def meter(self, ratio=True):
        from thexp import Meter

        meter = Meter()
        for key, offset in self.times.items():
            if ratio:
                if key == 'use':
                    continue
                meter[key] = offset / self.times['use']
            else:
                meter[key] = offset

        return meter

    def __str__(self):
        return pprint.pformat(self.times)

    def __getitem__(self, item):
        return self.times[item]

    def __getattr__(self, item):
        return self.times[item]


timeit = TimeIt()
