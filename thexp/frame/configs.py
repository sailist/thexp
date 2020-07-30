"""
程序运行期间全局配置，分为三个级别：

globals : 所有repo可以共用的配置
repository : 当前 git root 下的config
running：仅运行期间的临时配置
"""
from pprint import pformat
from typing import Any

from thexp.utils.paths import global_config, write_global_config
from ..globals import CONFIGL_, GITKEY_


class Globals:
    LEVEL = CONFIGL_

    def __init__(self):
        self._configs = [
            Config(CONFIGL_.running),
            Config(CONFIGL_.repository),
            Config(CONFIGL_.globals),
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

    def add_value(self, key, value, level=CONFIGL_.globals):
        if level == CONFIGL_.globals:
            self._configs[2][key] = value
        elif level == CONFIGL_.repository:
            self._configs[1][key] = value
        elif level == CONFIGL_.running:
            self._configs[0][key] = value
        else:
            assert False, 'level name error {}'.format(level)

    def get_value(self, key, level=CONFIGL_.globals, default=None):
        if level == CONFIGL_.globals:
            return self._configs[2][key]
        elif level == CONFIGL_.repository:
            return self._configs[1][key]
        elif level == CONFIGL_.running:
            return self._configs[0][key]
        else:
            assert False, 'level name error {}'.format(level)

    def items(self):
        return {
            CONFIGL_.globals: self._configs[2].items(),
            CONFIGL_.repository: self._configs[1].items(),
            CONFIGL_.running: self._configs[0].items(),
        }

    def __repr__(self):

        return "Globals({})".format(pformat({
            CONFIGL_.globals: self._configs[0].items(),
            CONFIGL_.repository: self._configs[1].items(),
            CONFIGL_.running: self._configs[2].items(),
        }))


class Config:
    """
    试验配置，根据等级分为用户级（整个用户），repo级（当前项目），实验级（当次运行）
    """
    config_levels = {CONFIGL_.running, CONFIGL_.repository, CONFIGL_.globals}

    def __init__(self, config_level):
        assert config_level in Config.config_levels, 'config level must in {}'.format(Config.config_levels)
        self._config_level = config_level
        if config_level == CONFIGL_.running:
            self._config_dict = {}
        elif config_level == CONFIGL_.repository:
            self._repo = None
            self._config_dict = None
        elif config_level == CONFIGL_.globals:
            self._config_dict = global_config()

    @property
    def repo(self):
        if self.config_level == CONFIGL_.repository:
            from ..utils.repository import load_repo
            self._repo = load_repo()
        return self._repo

    @property
    def repo_config(self):
        if not self.repo_level or self.repo is None:
            return {}
        if self._config_dict is None:
            from ..utils.repository import git_config
            self._config_dict = git_config(self.repo)
        return self._config_dict

    @property
    def config_level(self):
        return self._config_level

    @property
    def running_level(self):
        return self._config_level == CONFIGL_.running

    @property
    def globals_level(self):
        return self._config_level == CONFIGL_.globals

    @property
    def repo_level(self):
        return self._config_level == CONFIGL_.repository

    def __setitem__(self, key, value: str):
        """
        key 和 value尽量简洁
        :param key:
        :param value:
        :return:
        """
        key = str(key)
        if self._config_level == CONFIGL_.globals:
            self._config_dict[key] = value
            write_global_config(self._repo)
        elif self._config_level == CONFIGL_.repository:
            from thexp.utils.repository import git_config_syntax
            value = git_config_syntax(value)
            repo = self.repo
            if repo is not None:
                writer = repo.config_writer()
                writer.add_config(GITKEY_.section_name, key, value)
                writer.write()
                writer.release()
            self._config_dict[key] = value
        elif self._config_level == CONFIGL_.running:
            self._config_dict[key] = value

    def __getitem__(self, key):
        key = str(key)
        if self._config_level in {CONFIGL_.globals, CONFIGL_.running}:
            if key not in self._config_dict:
                raise AttributeError(key)
            return self._config_dict[key]
        elif self._config_level == CONFIGL_.repository:
            if key not in self.repo_config:
                raise AttributeError(key)
            return self.repo_config[key]

    def items(self):
        if self._config_level == CONFIGL_.running:
            return self._config_dict.items()
        elif self._config_level == CONFIGL_.repository:
            return self.repo_config.items()
        elif self._config_level == CONFIGL_.globals:
            return self._config_dict.items()

    def __contains__(self, item):
        try:
            _ = self[item]
            return True
        except:
            return False

    def __repr__(self) -> str:
        return "ExpConfig(level={}\nvalues={})".format(self._config_level, pformat(self.items()))


globs = Globals()
