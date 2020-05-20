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
import os
from collections import namedtuple
from datetime import datetime
from typing import Any

from ..utils.generel_util import curent_date
from ..utils.screen import ScreenStr

loginfo = namedtuple('loginfo', ['string', 'prefix_len'])


class Logger:
    VERBOSE = 0
    V_DEBUG = -1
    V_INFO = 0
    V_WARN = 1
    V_ERROR = 2
    V_FATAL = 3
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        if Logger._instance is not None:
            return Logger._instance
        return super().__new__(cls)

    def __init__(self, adddate=True, datefmt: str = '%y-%m-%d %H:%M:%S', sep: str = " | ", stdout=True):
        """
        :param datefmt:
        :param sep:
        """
        if Logger._instance is not None:
            return

        self.adddate = adddate
        self.datefmt = datefmt
        self.out_channel = []
        self.pipe_key = set()
        self.sep = sep
        self.return_str = ""
        self.listener = []
        self.stdout = stdout
        Logger._instance = self

    def format(self, *values, inline=False, fix=0, raw=False, append=False):
        """根据初始化设置 格式化 前缀和LogMeter"""
        if self.adddate and not raw:
            cur_date = datetime.now().strftime(self.datefmt)
        else:
            cur_date = ""

        values = ["{}".format(str(i)) for i in values]
        values = [i for i in values if len(i.strip()) != 0]

        if len(cur_date) == 0:
            space = [*["{}".format(str(i)) for i in values]]
        else:
            space = [cur_date, *values]

        if fix >= 0:
            left, right = self.sep.join(space[:fix + 1]), self.sep.join(space[fix + 1:])
            fix = len(left) + len(self.sep)
            logstr = self.sep.join((left, right))

            if inline:
                if append:
                    return "{}".format(logstr), fix
                else:
                    return "\r{}".format(logstr), fix
            else:
                return "{}\n".format(logstr), fix

        space = self.sep.join(space)

        if inline:
            return loginfo("\r{}".format(space), 0)
        else:
            return loginfo("{}\n".format(space), 0)

    def inline(self, *values, fix=0, append=False):
        """在一行内输出 前缀 和 LogMeter"""
        logstr, fix = self.format(*values, inline=True, fix=fix, append=append)
        self.handle(logstr, fix=fix)

    def info(self, *values, ):
        """以行为单位输出 前缀 和 LogMeter"""
        logstr, fix = self.format(*values, inline=False)
        self.handle(logstr, level=Logger.V_INFO, fix=fix)

    def raw(self, *values, inline=False, fix=0, level=0, append=False):
        """不输出日期前缀"""
        logstr, fix = self.format(*values, inline=inline, fix=fix, raw=True, append=append)
        self.handle(logstr, level=level)

    def debug(self, *values):
        logstr, fix = self.format("DEBUG", *values, inline=False)
        self.handle(logstr, level=Logger.V_DEBUG, fix=fix)

    def warn(self, *values):
        logstr, fix = self.format("WARN", *values, inline=False)
        self.handle(logstr, level=Logger.V_WARN, fix=fix)

    def error(self, *values):
        logstr, fix = self.format("ERROR", *values, inline=False)
        self.handle(logstr, level=Logger.V_ERROR, fix=fix)

    def fatal(self, *values):
        logstr, fix = self.format("FATAL", *values, inline=False)
        self.handle(logstr, level=Logger.V_FATAL, fix=fix)

    def newline(self):
        """换行"""
        self.handle("\n")

    def handle(self, logstr, end="", level=0, **kwargs):
        """
        handle log stinrg，以指定的方式输出
        :param logstr:
        :param _:
        :param end:
        :return:
        """
        for listener in self.listener:
            listener(logstr, end, level)

        if level < Logger.VERBOSE:
            return

        if logstr.startswith("\r"):
            fix = kwargs.get("fix", 0)
            self.return_str = logstr
            self.print(ScreenStr(logstr, leftoffset=fix), end=end)
        else:
            if len(self.return_str) != 0:
                self.print(self.return_str, end="\n")
            self.print(logstr, end="")

            for i in self.out_channel:
                with open(i, "a", encoding="utf-8") as w:
                    if len(self.return_str) != 0:
                        w.write("{}\n".format(self.return_str.strip()))
                    w.write(logstr)

            self.return_str = ""

    def print(self, *args, end='\n'):
        if self.stdout:
            print(*args, end=end, flush=True)

    def trig_stdout(self, val):
        self.stdout = val

    def add_log_dir(self, dir):
        """添加一个输出到文件的管道"""
        if dir in self.pipe_key:
            self.info("Add pipe {}, but already exists".format(dir))
            return None

        os.makedirs(dir, exist_ok=True)

        i = 0
        cur_date = curent_date(fmt="%y%m%d%H%M%S")
        fni = os.path.join(dir, "l.{}.{}.log".format(cur_date, i))
        while os.path.exists(fni):
            i += 1
            fni = os.path.join(dir, "l.{}.{}.log".format(cur_date, i))

        self.print("add output channel on {}".format(fni))
        self.out_channel.append(fni)
        self.pipe_key.add(dir)
        return fni

    def add_log_listener(self, func):
        self.listener.append(func)

    @staticmethod
    def set_verbose(verbose=0):
        Logger.VERBOSE = verbose
