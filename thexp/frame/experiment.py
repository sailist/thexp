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
import atexit
import json
import os
import re
import sys
import warnings
from collections import namedtuple
from pprint import pformat
from typing import Any

from ..globals import _CONFIGL, _GITKEY, _FNAME


def gitutils():
    from thexp.utils import gitutils
    return gitutils


from thexp.utils.generel_util import config_path, curent_date
from .params import BaseParams

'''
一个项目有多个实验，一个实验有多次试验，一次Python运行对应多次试验
实验放在一个文件夹，每次试验
'''
test_dir_pt = re.compile("[0-9]{4}\.[a-z0-9]{7}")


class Globals:
    LEVEL = _CONFIGL
    def __init__(self):
        self._git_config = ExpConfig(_CONFIGL.repository)
        self._configs = [
            ExpConfig(_CONFIGL.exp),
            self._git_config,
            ExpConfig(_CONFIGL.user),
        ]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, item):
        for config in self._configs:
            if item in config:
                return config[item]

    def __setitem__(self, key, value):
        self._configs[0][key] = value

    def add_value(self, key, value, level=_CONFIGL.exp):
        if level == _CONFIGL.exp:
            self._configs[0][key] = value
        elif level == _CONFIGL.user:
            self._configs[1][key] = value
        elif level == _CONFIGL.repository:
            self._configs[2][key] = value
        else:
            assert False, 'level name error {}'.format(level)

    def items(self):
        return {
            _CONFIGL.exp: self._configs[0].items(),
            _CONFIGL.repository: self._configs[1].items(),
            _CONFIGL.user: self._configs[2].items(),
        }

    def __repr__(self):

        return "Globals({})".format(pformat({
            _CONFIGL.exp: self._configs[0].items(),
            _CONFIGL.repository: self._configs[1].items(),
            _CONFIGL.user: self._configs[2].items(),
        }))


# repo.json 首先要知道所有的repo的位置信息，这或许可以通过为每个repo生成一个唯一的uuid来完成
# 每个repo都存在多个实验，每个实验做一次算一次试验
# 目录层级为 {实验名.repoid}/{试验名}/{...}

exception = namedtuple('exception_status', ['exc_type', 'exc_value', 'exc_tb'])

class Experiment:
    """
    用于目录管理和常量配置管理
    """
    user_level = _CONFIGL.user
    exp_level = _CONFIGL.exp
    repo_level = _CONFIGL.repository
    count = 0

    def __init__(self, exp_name):
        from git import Repo
        from ..utils.gitutils import ExpRepo

        self._exp_name = exp_name
        self._exp_dir = None
        self._start_time = None
        self._end_state = False  # 是否试验已结束
        self._test_dir = None
        self._project_dir = None
        self.git_config = ExpConfig(_CONFIGL.repository)
        self.exp_repo = ExpRepo.singleton()
        self.git_repo = self.exp_repo.repo  # type:Repo
        self._tags = {}
        self._hashs = []
        self._config = glob
        self._hold_dirs = []
        self._time_fmt = "%Y-%m-%d %H:%M:%S"
        self._plugins = {}
        self._exc_dict = None # type:exception
        self._initial()

    def add_config(self, key, value, level=_CONFIGL.exp):
        """
        添加配置
        :param key:
        :param value:
        :param level: 配置级别，查看 globals._CONFIGL
        :return:
        """
        self._config.add_value(key, value, level)

    def __getitem__(self, item):
        return self._config[item]

    def __setitem__(self, key, value):
        self._config[key] = value

    @property
    def commit(self):
        return self.exp_repo.commit

    @property
    def test_hash(self):
        if self.commit is None:
            return ""

        return self.commit.hexsha[:8]

    @property
    def commit_hash(self):
        if self.commit is None:
            return ""
        return self.commit.hexsha

    @property
    def project_name(self):
        return self[_GITKEY.projname]

    @property
    def root_dir(self):
        return self[_GITKEY.expsdir]

    @property
    def project_dir(self):
        if self._project_dir is None:
            self._project_dir = os.path.join(self.root_dir, '{}.{}'.format(self.project_name, self.exp_repo.uuid))
            os.makedirs(self._project_dir, exist_ok=True)

        return self._project_dir

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            self._exp_dir = os.path.join(self.project_dir, self._exp_name)
            os.makedirs(self._exp_dir, exist_ok=True)
        return self._exp_dir

    @property
    def test_dir(self):
        """
        获取当前 exp 目录下的 test_dir
        命名方式： 通过 序号.hash 的方式命名每一次实验
        会进行一系列判断保证在硬盘中的任何位置不会出现同名试验
        其中，hash 保证在不同时间运行的试验文件名不同，序号保证在同一进程运行的多次试验文件名不同
        {:04d}.{hash}

        Returns: 一个全局唯一的test_dir，(绝对路径)

        Notes:
            命名时还会确保存储路径下不会存在相同的序号（在生成时如果有相同序号则一直+1），方法为
            第一次获取试验目录时记录生成的序号，之后所有的序号都在此基础上生成，但如果 exp 目录变更，则
        """
        if self._test_dir is None:
            fs = os.listdir(self.exp_dir)

            if Experiment.count != 0: # 如果
                i = self.count + 1
            else:
                i = len([f for f in fs if re.search(test_dir_pt, f) is not None]) + 1

            cf_set = {f.split('.')[0] for f in fs} # 首位元素永远存在，不需要判断其他文件
            while "{:04d}".format(i) in cf_set:
                i += 1

            self._test_dir = os.path.join(self.exp_dir, "{:04d}.{}".format(i, self.test_hash))
            os.makedirs(self._test_dir, exist_ok=True)
            Experiment.count = i

        return self._test_dir

    @property
    def test_info_fn(self):
        return os.path.join(self.test_dir, _FNAME.info)

    @property
    def plugins(self):
        return self._plugins

    def makedir(self, name):
        """
        创建 test 级别的目录
        :param name: 目录名
        :return:  返回创建目录的绝对路径
        """
        d = os.path.join(self.test_dir, name)
        os.makedirs(d, exist_ok=True)
        self._hold_dirs.append(name)
        return d

    def make_exp_dir(self, name):
        """创建 exp 级别的目录"""
        d = os.path.join(self.exp_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def _initial(self):
        """初始化，会在实例被创建完成后调用"""
        if self._start_time is None:
            self._start_time = curent_date(self._time_fmt)
            sys.excepthook = self.exc_end
            atexit.register(self.end)

            gitutils().regist_exps(self.exp_repo, self._exp_name, self.exp_dir)
            self.exp_repo.check_commit()  # 在regist_exp 后运行，因为该方法可能导致 repo.json 文件修改
            self._write()
        else:
            warnings.warn("start the same experiment twice is not suggested.")

    def _write(self, **extra):
        """将试验状态写入试验目录中的 info.json """
        if self._end_state:
            return
        res = dict(
            repo=self.git_repo.working_dir,
            argv=sys.argv,
            exp_name=self._exp_name,
            exp_dir=self.exp_dir,
            test_name=os.path.basename(self.test_dir),
            test_dir=self.test_dir,
            root_dir=self.root_dir,
            project_name=self.project_name,
            project_iname=os.path.basename(self.project_dir),
            project_dir=self.project_dir,
            commit_hash=self.commit_hash,
            short_hash=self.test_hash,
            dirs=self._hold_dirs,
            time_fmt=self._time_fmt,
            start_time=self._start_time,
            tags=self._tags,
            plugins=self._plugins,
            **extra,
        )
        with open(self.test_info_fn, 'w', encoding='utf-8') as w:
            json.dump(res, w, indent=2)

    def end(self, end_code=0, **extra):
        """
        手动结束试验时调用该方法可以传入其他信息写入文件，
        该方法同时还通过 atexit 注册了退出钩子，如果不调用该方法，则在程序正常退出前会自动调用该方法写入结束文件。
        通过pycharm等IDE结束时，由于其结束机制不是通过抛出 KeyboardInterrupt 异常停止的，所以该方法存在没有正常调
        用的情况，此时文件中不会记录结束用时和退出状态码，也可以做为实验失败的判断条件。

        注意，结束状态仅会调用一次，无论是正常结束（对应方法end()）还是异常结束（对应方法exc_end()），
        在调用后的任何对实验的操作均不会写入到文件中
        :param end_code: 退出状态码，表明程序是否正常结束，以免引起数据混乱
        :param extra:
        :return:
        """
        self._write(
            end_time=curent_date(self._time_fmt),
            end_code=end_code,
            **extra
        )
        self._end_state = True

    def exc_end(self, exc_type, exc_val, exc_tb):
        """
        该方法注册了异常钩子，无需手动调用，在出现异常且没有处理导致程序退出的情况时，会通过该方法记录程序失败原因，
        包括异常类型和提示，同时会将异常栈内容输出到试验目录下的 Exception 文件中
        :param exc_type:  异常传入类型
        :param exc_val:  异常传入示例
        :param exc_tb:  traceback
        :return:
        """
        import traceback
        with open(os.path.join(self.test_dir, _FNAME.Exception), 'w', encoding='utf-8') as w:
            w.write("".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
        self.end(
            end_code=1,
            exc_type=traceback.format_exception_only(exc_type, exc_val)[-1].strip()
        )
        self._exc_dict = exception(exc_tb, exc_val, exc_tb)
        self._end_state = True
        traceback.print_exception(exc_type, exc_val, exc_tb)

    def config_items(self):
        """
        三个等级的配置文件的内容
        :return:
        """
        return self._config.items()

    def add_tag(self, name, *plugins):
        """
        插件的插件，用来表示tag对应的某个插件利用另一个插件做了某些什么事情
        :param name:
        :param plugins:
        :return:
        """
        self._tags[name] = plugins
        self._write()

    def regist_plugin(self, key, value=None):
        """
        注册试验中使用的插件。
        :param key: 插件名
        :param value: 若为空，则为空字典
        :return:
        """
        if value is None:
            value = {}
        self._plugins[key] = value
        self._write()

    def regist_exit_hook(self, func):
        """
        提供给用户的额外注册退出钩子，传入的参数有两个，
        第一个参数为该 Experiment 的实例对象
        第二个参数为一个 exception_status 实例，当程序的退出是因为某异常引起时，该参数会被传入
            exception_status 是一个 namedtuple('exception_status', ['exc_type', 'exc_value', 'exc_tb'])
            可以通过 Python 内置的 traceback 库来处理，如

            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
        :param func: 退出前调用的钩子，
        :return:
        """
        import atexit
        def exp_func():
            func(self,self._exc_dict)
        atexit.register(exp_func)


class ExpConfig:
    """
    试验配置，根据等级分为用户级（整个用户），repo级（当前项目），实验级（当次运行）
    """
    config_levels = {_CONFIGL.user, _CONFIGL.repository, _CONFIGL.exp}

    def __init__(self, config_level):
        assert config_level in ExpConfig.config_levels, 'config level must in {}'.format(ExpConfig.config_levels)
        self._config_level = config_level
        if config_level == _CONFIGL.user:
            self._repo = self.load_user_config()
        elif config_level == _CONFIGL.repository:
            self._repo = None
        elif config_level == _CONFIGL.exp:
            self._repo = BaseParams()

    @property
    def repo(self):
        from ..utils.gitutils import ExpRepo
        if self.config_level == _CONFIGL.repository:
            return ExpRepo().repo
        return self._repo

    @property
    def config_level(self):
        return self._config_level

    @property
    def user_level(self):
        return self._config_level == _CONFIGL.user

    @property
    def exp_level(self):
        return self._config_level == _CONFIGL.exp

    @property
    def repo_level(self):
        return self._config_level == _CONFIGL.repository

    def __setitem__(self, key, value: str):
        """
        key 和 value尽量简洁
        :param key:
        :param value:
        :return:
        """
        key = str(key)
        if self._config_level == _CONFIGL.user:
            self._repo[key] = value
            with open(config_path(), "w") as w:
                json.dump(self._repo, w)
        elif self._config_level == _CONFIGL.repository:
            value = gitutils().git_config_syntax(value)
            writer = self._repo.config_writer()
            writer.add_config(_GITKEY.section_name, key, value)
            writer.write()
            writer.release()
        elif self._config_level == _CONFIGL.exp:
            self._repo[key] = value

    def __getitem__(self, key):
        key = str(key)
        if self._config_level == _CONFIGL.user:
            if key not in self.repo:
                raise AttributeError(key)
            return self.repo[key]
        elif self._config_level == _CONFIGL.repository:
            if self.repo is None:
                raise AttributeError(key)
            reader = self.repo.config_reader()
            res = reader.get_value(_GITKEY.section_name, key, None)
            if res is None:
                raise AttributeError(key)
            return res
        elif self._config_level == _CONFIGL.exp:
            if key not in self.repo:
                raise AttributeError(key)
            return self.repo[key]

    def items(self):
        if self._config_level == _CONFIGL.user:
            return self.repo.items()
        elif self._config_level == _CONFIGL.repository:
            if self.repo is None:
                return {}
            reader = self.repo.config_reader()

            reader.get_value(_GITKEY.thexp, _GITKEY.projname)
            return reader.items(_GITKEY.section_name)
        elif self._config_level == _CONFIGL.exp:
            return self.repo.items()

    def __contains__(self, item):
        try:
            res = self[item]
            return True
        except:
            return False

    def load_user_config(self):
        with open(config_path(), "r") as r:
            return json.load(r)

    def __repr__(self) -> str:
        return "ExpConfig(level={}\nvalues={})".format(self._config_level, pformat(self.items()))


glob = Globals()
