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
import json
import os
import re
import sys
import warnings
from typing import Any
from pprint import pformat



from ..globals import _CONFIGL, _GITKEY, _FNAME


def gitutils():
    from thexp.utils import gitutils
    gitutils.check_commit_exp()
    return gitutils


from thexp.utils.generel_util import config_path, curent_date
from .params import BaseParams

'''
一个项目有多个实验，一个实验有多次试验，一次Python运行对应多次试验
实验放在一个文件夹，每次试验
'''
test_dir_pt = re.compile("[0-9]{4}\.[a-z0-9]{7}")


class Globals:
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

        assert False, 'level name error'

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

class Experiment:
    """
    用于目录管理和常量配置管理
    """
    user_level = _CONFIGL.user
    exp_level = _CONFIGL.exp
    repo_level = _CONFIGL.repository

    def __init__(self, exp_name):
        from git import Repo

        self._exp_name = exp_name
        self._exp_dir = None
        self._start_time = None
        self._end_state = None
        self._test_dir = None
        self._project_dir = None
        self.git_config = ExpConfig(_CONFIGL.repository)
        self.git_repo = gitutils().repo  # type:Repo
        self._tags = {}
        self._config = glob
        self._hold_dirs = []
        self._time_fmt = "%y-%m-%d-%H%M%S"
        self._plugins = {}

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
        gitutils().check_commit_exp()
        return gitutils().commit

    @property
    def test_hash(self):
        if self.commit is None:
            return ""

        return self.commit.hexsha[:8]

    @property
    def commit_hash(self):
        if self.commit is None:
            return ""
        return gitutils().commit.hexsha

    @property
    def project_name(self):
        return self[_GITKEY.projname]

    @property
    def root_dir(self):
        return self[_GITKEY.expsdir]

    @property
    def project_dir(self):
        if self._project_dir is not None:
            return self._project_dir
        self._project_dir = os.path.join(self.root_dir, '{}.{}'.format(self.project_name,gitutils().uuid))
        os.makedirs(self._project_dir, exist_ok=True)
        return self._project_dir

    @property
    def exp_dir(self):
        if self._exp_dir is not None:
            return self._exp_dir
        self._exp_dir = os.path.join(self.project_dir, self._exp_name)
        os.makedirs(self._exp_dir, exist_ok=True)
        return self._exp_dir

    @property
    def test_dir(self):
        if self._test_dir is not None:
            return self._test_dir
        fs = os.listdir(self.exp_dir)

        i = len([i for i in fs if re.search(test_dir_pt, i) is not None]) + 1
        self._test_dir = os.path.join(self.exp_dir, "{:04d}.{}".format(i, self.test_hash))
        os.makedirs(self._test_dir, exist_ok=True)

        return self._test_dir

    @property
    def test_info_fn(self):
        return os.path.join(self.test_dir, _FNAME.info)

    def makedir(self, name):
        d = os.path.join(self.test_dir, name)
        os.makedirs(d, exist_ok=True)
        self._hold_dirs.append(name)
        return d

    def make_exp_dir(self, name):
        d = os.path.join(self.exp_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def start(self):
        if self._start_time is None:
            self._start_time = curent_date(self._time_fmt)
            sys.excepthook = self.exc_end
            gitutils().regist_exps(self._exp_name,self.exp_dir)
            self._write()
        else:
            warnings.warn("start the same experiment twice is not suggested.")

    def _write(self, **extra):
        res = dict(
            repo=gitutils().repo.working_dir,
            argv=sys.argv,
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
        :param end_code:
        :param extra:
        :return:
        """
        self._write(
            end_time=curent_date(self._time_fmt),
            end_code=end_code,
        )

    def exc_end(self, exc_type, exc_val, exc_tb):
        import traceback
        with open(os.path.join(self.test_dir, _FNAME.Exception), 'w', encoding='utf-8') as w:
            w.write("".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
        self.end(
            end_code=1,
            exc_type=traceback.format_exception_only(exc_type, exc_val)[-1].strip()
        )
        traceback.print_exception(exc_type, exc_val, exc_tb)

    def config_items(self):
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

    def regist_plugin(self, key, value=""):
        self._plugins[key] = value
        self._write()

    def regist_exit_hook(self, func):
        import atexit
        atexit.register(func)


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
        if self.config_level == _CONFIGL.repository:
            return gitutils().repo
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
            writer.add_config(gitutils().section_name, key, value)
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
            res = reader.get_value(gitutils().section_name, key, None)
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
            # reader.items(section_name)
            reader.get_value(_GITKEY.thexp, _GITKEY.projname)
            return reader.items(gitutils().section_name)
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
        return "ExpConfig(level={}\nvalues={})".format(self._config_level,pformat(self.items()))


glob = Globals()
