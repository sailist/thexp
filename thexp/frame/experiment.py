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

from git import Repo


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
    def __init__(self):
        self._git_config = ExpConfig("repository")
        self._configs = [
            ExpConfig("exp"),
            self._git_config,
            ExpConfig("user"),
        ]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self.__setitem__(name,value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, item):
        for config in self._configs:
            if item in config:
                return config[item]

    def __setitem__(self, key, value):
        self._configs[0][key] = value

    def add_value(self, key, value, level='exp'):
        if level == 'exp':
            self._configs[0][key] = value
        elif level == 'user':
            self._configs[1][key] = value
        elif level == 'repository':
            self._configs[2][key] = value

        assert False, 'level name error'

    def items(self):
        return {
            'exp':self._configs[0].items(),
            'repository':self._configs[1].items(),
            'user':self._configs[2].items(),
        }

class Experiment:
    """
    用于目录管理和常量配置管理
    """
    user_level = 'user'
    exp_level = 'exp'
    repo_level = 'repository'

    def __init__(self, exp_name):
        self._exp_name = exp_name
        self._exp_dir = None
        self._start_time = None
        self._end_state = None
        self._test_dir = None
        self._project_dir = None
        self.git_config = ExpConfig("repository")
        self.git_repo = gitutils().repo  # type:Repo

        self._config = glob
        self._hold_dirs = []

    def add_value(self,key,value,level='exp'):
        self._config.add_value(key,value,level)

    def __getitem__(self, item):
        return self._config[item]

    def __setitem__(self, key, value):
        self._config[key] = value

    @property
    def commit(self):
        gitutils().check_commit_exp()
        return gitutils().commit

    @property
    def project_name(self):
        return self["projname"]

    @property
    def test_hash(self):
        return self.commit.hexsha[:8]

    @property
    def root_dir(self):
        return self['expsdir']

    @property
    def project_dir(self):
        if self._project_dir is not None:
            return self._project_dir
        self._project_dir = os.path.join(self.root_dir, self.project_name)
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
        return os.path.join(self.test_dir, "info.json")

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
            self._start_time = curent_date()
            gitutils().check_commit_exp()
            sys.excepthook = self.exc_end
        else:
            warnings.warn("start the same experiment twice is not suggested.")

    def end(self, end_code=0, **extra):
        """
        :param end_code:
        :param extra:
        :return:
        """
        res = dict(
            commit_hash=gitutils().commit.hexsha,
            short_hash=self.test_hash,
            exp_dir=self.test_dir,
            dirs=self._hold_dirs,
            start_time=gitutils().commit_time,
            end_time=curent_date(),
            end_code=end_code,
            **extra,
        )
        with open(self.test_info_fn, 'w', encoding='utf-8') as w:
            json.dump(res, w, indent=2)

    def exc_end(self, exc_type, exc_val, exc_tb):
        import traceback
        with open(os.path.join(self.test_dir,"Exception"),'w',encoding='utf-8') as w:
            w.write("".join(traceback.format_exception(exc_type,exc_val,exc_tb)))
        self.end(
            end_code=1,
            exc_type=traceback.format_exception_only(exc_type,exc_val)[-1].strip()
        )
        traceback.print_exception(exc_type,exc_val,exc_tb)

    def config_items(self):
        return self._config.items()





class ExpConfig:
    """
    试验配置，根据等级分为用户级（整个用户），repo级（当前项目），实验级（当次运行）
    """
    config_levels = {"user", "repository", 'exp'}

    def __init__(self, config_level):
        assert config_level in ExpConfig.config_levels, 'config level must in "user", "repository"'
        self._config_level = config_level
        if config_level == "user":
            self._repo = self.load_user_config()
        elif config_level == "repository":
            self._repo = gitutils().repo
        elif config_level == "exp":
            self._repo = BaseParams()

    @property
    def repo(self):
        return self._repo

    @property
    def config_level(self):
        return self._config_level

    @property
    def user_level(self):
        return self._config_level == "user"

    @property
    def exp_level(self):
        return self._config_level == "exp"

    @property
    def repo_level(self):
        return self._config_level == 'repository'

    def __setitem__(self, key, value: str):
        """
        key 和 value尽量简洁
        :param key:
        :param value:
        :return:
        """
        key = str(key)
        if self._config_level == "user":
            self._repo[key] = value
            with open(config_path(), "w") as w:
                json.dump(self._repo, w)
        elif self._config_level == "repository":
            value = gitutils().git_config_syntax(value)
            writer = self._repo.config_writer()
            writer.add_value(gitutils().section_name, key, value)
            writer.write()
            writer.release()
        elif self._config_level == "exp":
            self._repo[key] = value

    def __getitem__(self, key):
        key = str(key)
        if self._config_level == "user":
            if key not in self._repo:
                raise AttributeError(key)
            return self._repo[key]
        elif self._config_level == "repository":
            reader = self._repo.config_reader()
            res = reader.get_value(gitutils().section_name, key, None)
            if res is None:
                raise AttributeError(key)
            return res
        elif self._config_level == "exp":
            if key not in self._repo:
                raise AttributeError(key)
            return self._repo[key]

    def items(self):
        if self._config_level == "user":
            return self._repo.items()
        elif self._config_level == "repository":
            reader = self._repo.config_reader()
            # reader.items(section_name)
            reader.get_value('thexp','projname')
            return reader.items(gitutils().section_name)
        elif self._config_level == "exp":
            return self._repo.items()


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
        return super().__repr__()


glob = Globals()