import functools
import time
import torch
import os
import subprocess
import re
from thexp.base_classes.tree import tree
from functools import partial

match_mem = re.compile('([0-9]+) +([0-9]+)[^|]* ([0-9]+)MiB')


class DeviceMem():
    def __init__(self):
        self.line_mem = tree()

    def try_parse(self, lines, pid, device):
        lid = self.line_mem[pid][device]
        if isinstance(lid, dict):
            return -1
        elif lid > len(lines):
            return -1
        else:
            res = re.search(match_mem, lines[lid].decode())
            if res is None:
                return -1
            else:
                _device, _pid, _mib = [int(i) for i in res.groups()]
                if _pid == pid and _device == device:
                    return _mib
                else:
                    return -1

    def re_parse(self, lines, pid, device):
        for lid, line in enumerate(lines):
            res = re.search(match_mem, line.decode())
            if res is not None:
                _device, _pid, _mib = [int(i) for i in res.groups()]
                if _pid == pid and _device == device:
                    self.line_mem[pid][device] = lid
                    return _mib
        return 0

    def get_pid_device_mem(self, pid, device):
        """
        尽可能有效率的得到进程在某设备下占用的显存（通过命令行程序调用获取）
        :param pid:
        :param device:
        :return:
        """
        if isinstance(device, torch.device):
            device = device.index
        elif isinstance(device, str):
            device = torch.device(device).index

        proc = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
        lines = proc.stdout.readlines()
        mib = self.try_parse(lines, pid, device)
        if mib == -1:
            mib = self.re_parse(lines, pid, device)

        return mib


_memer = DeviceMem()
_pid = os.getpid()

get_pid_device_mem = partial(_memer.get_pid_device_mem, pid=_pid, device=torch.cuda.current_device())



class memory(object):
    r"""
    便捷抢卡
    Args:
        memory: 需要占用的内存，以 MB 为单位
        device: 需要占用内存的设备
        hold:
        unit:
    Example::
        >>> import thexp
        >>> with thexp.memory(5000):
        ...   y = x * 2

        >>> @thexp.memory(1024)
        ... def doubler(x):
        ...     ...

        >>> thexp.memory(10000).start()
        ... # do something

    Why use nvidia-smi to get memory useage? see:
        https://github.com/pytorch/pytorch/issues/12873
    """

    def __init__(self, memory, device=None, hold=False, unit=5) -> None:
        super().__init__()
        if device is None:
            device = torch.cuda.current_device()
        if isinstance(device, (str, int)):
            device = torch.device(device)

        self.need = memory
        self.device = device
        self.hold = hold
        self.unit = unit
        self.exc_time = 0
        self.acc = 0
        self.mem = []
        self.last_success = _memer.get_pid_device_mem(_pid, self.device)

    def start(self):
        try:
            while self.last_success < self.need:
                try:
                    tmp = torch.rand([self.unit + self.acc, 1048576], device=self.device)
                    self.mem.append(tmp)
                    self.acc += self.unit
                    self.last_success = _memer.get_pid_device_mem(_pid, self.device)
                    time.sleep(0.1)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        pass
                    else:
                        print(e)
                    self.exc_time += 1
                    self.acc = max(self.acc - self.unit, 0)
                    time.sleep(0.5)
                print('{}/{}Mb, try={}, pid={}'.format(self.last_success,
                                                       self.need,
                                                       self.exc_time,
                                                       os.getpid()), end='\r')
            print()
            if self.hold:
                input('any input...')
            else:
                [i.cpu() for i in self.mem]
                del self.mem[:]
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print('\nabort.')

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        del self.mem[:]
        torch.cuda.empty_cache()
        return True

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad

    @staticmethod
    def hold_current():
        count = torch.cuda.device_count()
        mems = [_memer.get_pid_device_mem(_pid, i) for i in range(count)]
        for i, mem in enumerate(mems):
            if mem > 0:
                memory(mem, device=i, hold=(i == count - 1)).start()
