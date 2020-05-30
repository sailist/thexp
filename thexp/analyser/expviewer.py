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
import pprint
from collections.abc import Iterable
from functools import lru_cache
from typing import Union, List

from thexp.base_classes.attr import attr
from thexp.globals import _INFOJ, _FNAME, _REPOJ, _INDENT, _BUILTIN_PLUGIN, _DLEVEL
from thexp.utils.generel_util import date_from_str, home_dir
from ..utils.generel_util import deep_chain


class SummaryViewer():
    def __init__(self):
        self._repos_info = None

    @property
    def repos_info(self) -> dict:
        """
        一个repos_info 的格式如下，该文件以json格式保存在 home/.thexp/repo.json

        内部以键值对形式保存了所有项目的名称、路径、该项目下的所有实验
        {
          "thexp.ce": {
            "repopath": "E:\\Python\\iLearn\\thexp",
            "exps": {
              "DemoExp.__main__": "e:/python/ilearn/thexp/.thexp/experiments\\thexp.ce\\DemoExp.__main__",
              "myExp.__main__": "e:/python/ilearn/thexp/.thexp/experiments\\thexp.ce\\myExp.__main__"
            }
          }
        }
        :return:
        """

        if self._repos_info is not None:
            return self._repos_info

        rp = os.path.join(home_dir(), _FNAME.repo)
        if not os.path.exists(rp):
            self._repos_info = {}
        else:
            with open(rp, encoding='utf-8') as r:
                self._repos_info = json.load(r)
        return self._repos_info

    def clean(self):
        """
        清除所有项目中不存在的实验
        :return:
        """
        res = {}
        for k, v in self.repos_info.items():
            exps = {k: v for k, v in v[_REPOJ.exps].items() if os.path.exists(v)}
            v[_REPOJ.exps] = exps
            res[k] = v
        rp = os.path.join(home_dir(), _FNAME.repo)
        with open(rp, 'w', encoding='utf-8') as w:
            json.dump(res, w)
        self.refresh()

    def refresh(self):
        """刷新repos_info"""
        self._repos_info = None

    @property
    def repos(self):
        """获取所有的项目名称列表"""
        return list(self.repos_info.keys())

    @property
    def proj_names(self):
        return list(self.repos_info.keys())

    @property
    def repopaths(self):
        """获取所有的项目路径"""
        return [i[_REPOJ.repopath] for i in self.repos_info.values()]

    @property
    def proj_viewers(self):
        res = []
        for repopath, repo_name in zip(self.repopaths, self.repos):
            pve = ProjViewer(repopath)
            if repo_name.split('.')[-1] == pve.hash:
                res.append(pve)
        return res

    def all(self):
        """获取所有试验的TestViewer对象"""
        from ..base_classes.list import llist
        return llist(deep_chain([i.all() for i in self.proj_viewers]))

    def tree(self, depth=None):
        """
        按照树结构列出所有试验，方便回溯信息
        Args:
            depth: 查找深度，最大为3 具体的。
                0：输出本地的项目名
                1：本地项目各信息及实验名
                2：本地项目、实验详细信息及试验名
                3/-1：包括试验详细信息在内的完全信息

        Returns:

        """
        if depth == 0:
            return [k for k in self.repos_info.keys()]

        if depth is not None:
            depth -= 1

        res = {}
        for k, v in self.repos_info.items():
            if os.path.exists(v[_REPOJ.repopath]):
                pve = ProjViewer(v[_REPOJ.repopath])
                # 判断一个项目是否清空了 git，如果不清空git，由于Projviewer内没有办法得知原hash，
                # 而repo.json 内又没有办法更新，所以会导致重复获取
                if pve.hash == k.split('.')[-1]:
                    res[k] = ProjViewer(v[_REPOJ.repopath]).tree(depth)
        return res

    def find(self,
             *names,
             level=_DLEVEL.test) -> Union[List['TestViewer'], List['ProjViewer'], List['ExpViewer']]:
        """
        根据 names  获取对应的 ProjViewer / ExpViewer / TestViewer
        Args:
            *names:
            level: _DLEVEL

        Returns:

        """
        if level == _DLEVEL.proj:
            return [i for i in self.proj_viewers if i.proj_name in names]
        elif level == _DLEVEL.exp:
            return list(deep_chain([i.find(*names, level) for i in self.proj_viewers]))
        return list(deep_chain([i.find(*names) for i in self.proj_viewers]))

    def __repr__(self):
        return "SummaryViewer(repos=[{}])".format(self.repos)

    def __getitem__(self, item):
        if isinstance(item, dict):
            res = []
            for key in item:
                if isinstance(key, int):
                    res.append(ProjViewer(self.repopaths[key])[item[key]])
                elif isinstance(key, str):
                    try:
                        r = self.proj_names.index(key)
                        res.append(ProjViewer(self.repopaths[r])[item[key]])
                    except:
                        raise AttributeError(key)
                else:
                    raise AttributeError(key)
            return list(deep_chain(res))

        if isinstance(item, slice):
            return self.proj_viewers[item]
        elif isinstance(item, Iterable):
            return [self.proj_viewers[i] for i in item]
        elif isinstance(item, int):
            return [self.proj_viewers[item]]
        else:
            raise AttributeError(item)


class ProjViewer():
    """
    以项目为单位的实验浏览类，用于查看所有 未删除 的 exp 对应的方法
    """

    def __init__(self, repo_dir):
        self.repo_dir = repo_dir
        self._repo = None
        self._repo_info = None

    @property
    @lru_cache()
    def repo(self):
        from ..utils.gitutils import Repo
        return Repo(self.repo_dir)

    @property
    def proj_name(self):
        ks = list(self.repo_info.keys())
        if len(ks) == 0:
            return None

        return list(self.repo_info.keys())[0]

    @property
    def hash(self):
        if self.proj_name is None:
            return ""
        return self.proj_name.split('.')[-1]

    @property
    def exp_root(self):
        return self.repo_info[self.proj_name]['exp_root']

    @property
    def repo_info(self):
        if self._repo_info is not None:
            return self._repo_info

        fn = os.path.join(self.repo_dir, _FNAME.repo)
        if os.path.exists(fn):
            with open(fn) as r:
                self._repo_info = json.load(r)
        else:
            self._repo_info = {}
        return self._repo_info

    @property
    def exps(self):
        if self.proj_name is None:
            return []
        return [v for v in self.exp_dict.values()]

    @property
    def exp_names(self):
        if self.proj_name is None:
            return []
        return [k for k in self.exp_dict.keys()]

    @property
    def exp_dict(self) -> dict:
        if self.proj_name is None:
            return {}
        res = {}
        for k in self.repo_info[self.proj_name]['exps']:
            fn = os.path.join(self.exp_root, self.proj_name, k)
            if os.path.exists(fn):
                res[k] = fn
        return res

    @property
    def exp_viewers(self):
        return [ExpViewer(i) for i in self.exps]

    def find(self, *names, level=_DLEVEL.test) -> Union[List['ExpViewer'], List[List['TestViewer']]]:
        if level == _DLEVEL.exp:
            return [i for i in self.exp_viewers if i.exp_name in names]
        return [i.find(*names) for i in self.exp_viewers]

    def tree(self, depth=None):
        if depth == 0:
            return {
                "repopath": self.repo_dir,
                "exps": [k for k in self.exp_names]
            }
        else:
            if depth is not None:
                depth -= 1

            return {
                "repopath": self.repo_dir,
                "exps": {k: ExpViewer(v).tree(depth) for k, v in self.exp_dict.items()}
            }

    def all(self):
        return [i.all() for i in self.exp_viewers]

    def filter(self, condition, to_end=True, include_hide=False):
        if to_end:
            return list(deep_chain([i.all(include_hide=include_hide) for i in self.exp_viewers if condition(i)]))
        return [i for i in self.exp_viewers if condition(i)]

    def __repr__(self):
        return "ProjViewer(exps=[{}])".format(self.exp_names)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        if isinstance(item, dict):
            res = []
            for key in item:
                if isinstance(key, int):
                    res.append(ExpViewer(self.exps[key])[item[key]])
                elif isinstance(key, str):
                    try:
                        r = self.exp_names.index(key)
                        res.append(ExpViewer(self.exps[r])[item[key]])
                    except:
                        raise AttributeError(key)
                else:
                    raise AttributeError(key)
            return res

        if isinstance(item, slice):
            return [ExpViewer(i) for i in self.exps[item]]
        elif isinstance(item, Iterable):
            return [ExpViewer(self.exps[i]) for i in item]
        elif isinstance(item, int):
            return [ExpViewer(self.exps[item])]
        raise AttributeError(item)


class ExpViewer():
    """
    :param trainer_cls: cls of trainer or str
    """

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self._show_dirs = None
        self.exp_name = os.path.basename(exp_dir)

    @property
    def show_dirs(self):
        if self._show_dirs is None:
            if not os.path.exists(self.exp_dir):
                self._show_dirs = []
            else:
                fs = [os.path.join(self.exp_dir, f) for f in os.listdir(self.exp_dir)]
                fs = [f for f in fs if
                      os.path.exists(os.path.join(f, 'info.json')) and not os.path.exists(os.path.join(f, _FNAME.hide))]
                self._show_dirs = fs

        return self._show_dirs

    @property
    def hide_dirs(self):
        if not os.path.exists(self.exp_dir):
            return []
        fs = [os.path.join(self.exp_dir, f) for f in os.listdir(self.exp_dir)]
        fs = [f for f in fs if
              os.path.exists(os.path.join(f, 'info.json')) and os.path.exists(os.path.join(f, _FNAME.hide))]

        return fs

    @property
    def test_dirs(self):
        if not os.path.exists(self.exp_dir):
            return []
        fs = [os.path.join(self.exp_dir, f) for f in os.listdir(self.exp_dir)]
        fs = [f for f in fs if os.path.exists(os.path.join(f, 'info.json'))]
        return fs

    @property
    def test_names(self):
        return [os.path.basename(i) for i in self.test_dirs]

    def test_viewers(self):
        """默认返回所有可见的 试验 的 TestViewer 对象"""
        return [TestViewer(f) for f in self.test_names]

    def hide_all(self):
        self._show_dirs = []
        hide_tests(*self.show_dirs)

    def show_all(self):
        show_tests(*self.hide_dirs)
        self._show_dirs = None

    def find(self, *test_names):
        return [TestViewer(test_dir) for test_dir, test_name in zip(self.test_dirs, self.test_names) if
                test_name in test_names]

    def all(self, include_hide=False):
        if include_hide:
            return [TestViewer(i) for i in self.test_dirs]
        else:
            return [TestViewer(i) for i in self.show_dirs]

    def tree(self, depth=None):
        if depth == 0:
            return {
                'exp_name': self.exp_name,
                'exp_path': self.exp_dir,
                'tests': [os.path.basename(k) for k in self.show_dirs]
            }
        return {
            'exp_name': self.exp_name,
            'exp_path': self.exp_dir,
            'tests': {
                os.path.basename(k): TestViewer(k).tree() for k in self.show_dirs
            }
        }

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [TestViewer(i) for i in self.show_dirs[item]]
        elif isinstance(item, str):
            return [TestViewer(self.test_dirs[self.test_names.index(item)])]
        elif isinstance(item, Iterable):
            return [self[i] for i in item]
        elif isinstance(item, int):
            return [TestViewer(self.show_dirs[item])]
        assert AttributeError(item)

    def __repr__(self):
        return "ExpViewer({} tests)".format(len(self.show_dirs))


class TestViewer():
    def __init__(self, test_dir: str):
        self.test_dir = test_dir

        self.c, self.commithash = os.path.basename(test_dir).split(".")
        self._json_info = None

    def show(self):
        res = show_tests(self.test_dir)

    def hide(self):
        res = hide_tests(self.test_dir)

    def fav(self):
        res = fav(self.test_dir)

    def unfav(self):
        res = unfav(self.test_dir)

    @property
    def commit(self):
        return None

    @property
    @lru_cache()
    def repo(self):
        from ..utils.gitutils import Repo

        return Repo(self.repopath)

    @property
    def test_name(self):
        return self.json_info[_INFOJ.test_name]

    @property
    def json_fn(self):
        return os.path.join(self.test_dir, _FNAME.info)

    @property
    def json_info(self) -> dict:
        if self._json_info is None:
            if not os.path.exists(self.json_fn):
                self._json_info = attr()
            with open(self.json_fn, encoding='utf-8') as r:
                self._json_info = attr(json.load(r))
        self._json_info["test_dir"] = self.test_dir
        self._json_info['visible'] = self.visible
        self._json_info['fav'] = self.fav_state
        return self._json_info.jsonify()

    @property
    def visible(self):
        return not os.path.exists(os.path.join(self.test_dir, _FNAME.hide))

    @property
    def fav_state(self):
        return os.path.exists(os.path.join(self.test_dir, _FNAME.fav))

    def tree(self):
        return self.json_info

    @property
    def repopath(self):
        return self.json_info[_INFOJ.repo]

    @property
    def argv(self):
        return self.json_info[_INFOJ.argv]

    @property
    def commit_hash(self):
        return self.json_info[_INFOJ.commit_hash]

    @property
    def success_exit(self):
        return _INFOJ.end_code in self.json_info and self.json_info[_INFOJ.end_code] == 0

    @property
    def start_time(self):
        return date_from_str(self.json_info[_INFOJ.start_time], self.json_info[_INFOJ.time_fmt])

    @property
    def end_time(self):
        return date_from_str(self.json_info[_INFOJ.end_time], self.json_info[_INFOJ.time_fmt])

    @property
    @lru_cache()
    def board_reader(self):
        from .reader import BoardReader
        from ..globals import _PLUGIN_WRITER

        writer = self.get_plugin(_BUILTIN_PLUGIN.writer)
        if writer is None:
            return None

        fs = os.listdir(writer[_PLUGIN_WRITER.log_dir])
        fs = [i for i in fs if i.endswith('.bd') and i.startswith('events.out.tfevents')]
        if len(fs) == 0:
            return None
        if len(fs) > 1:
            import warnings
            warnings.warn('found multi tfevents file, return the first one')
        file = os.path.join(writer[_PLUGIN_WRITER.log_dir], fs[0])

        return BoardReader(file)

    @property
    @lru_cache()
    def metrics(self):
        from .compare import Metric
        res = {}
        for stag in self.board_reader.scalars_tags:
            values = self.board_reader.get_scalars(stag).values
            metric = Metric.guess(stag, values)
            v = eval(metric.type)(values)
            res[metric.key] = v
        return res

    @property
    @lru_cache()
    def params(self):
        from ..frame.params import BaseParams
        if self.has_plugin(_BUILTIN_PLUGIN.params):
            p = BaseParams().from_json(os.path.join(self.test_dir, _FNAME.params))
        else:
            p = BaseParams()
        return p

    @property
    def duration(self):
        return self.end_time - self.start_time

    def has_tag(self, tag: str):
        return _INFOJ.tags in self.json_info and tag in self.json_info[_INFOJ.tags]

    def has_dir(self, dir: str):
        return _INFOJ.dirs in self.json_info and dir in self.json_info[_INFOJ.dirs]

    def has_plugin(self, plugin: str):
        return _INFOJ.plugins in self.json_info and plugin in self.json_info[_INFOJ.plugins]

    def get_plugin(self, plugin: str):
        if self.has_plugin(plugin):
            return self.json_info[_INFOJ.plugins][plugin]
        return None

    def get_tag(self, tag: str):
        if self.has_tag(tag):
            return self.json_info[_INFOJ.tags][tag]
        return None

    @property
    def plugins(self) -> dict:
        dic = self.json_info[_INFOJ.plugins]
        for tag, plugins in self.json_info[_INFOJ.tags].items():
            for plugin in plugins:
                if plugin not in dic:
                    dic[plugin] = {
                        _INFOJ._tags: []
                    }
                else:
                    lis = dic[plugin].setdefault(_INFOJ._tags, [])
                    lis.append(tag)
        return dic

    @property
    def tags(self):
        dic = dict()
        for tag, plugins in self.json_info[_INFOJ.tags].items():
            if len(plugins) == 0:
                plugins = [_INFOJ._]
            for plugin in plugins:
                lis = dic.setdefault(plugin, [])
                lis.append(tag)
        return dic

    def summary(self):
        from ..frame.meter import Meter
        import textwrap
        m = Meter()
        m['exit status'] = 1 - int(self.success_exit)
        m['start'] = self.start_time
        m['end'] = self.end_time
        m['duration'] = self.duration
        print(m)
        print('tags:')
        for k, v in self.tags.items():
            print(textwrap.indent('{} : {}'.format(k, pprint.pformat(v)), _INDENT.ttab))
        print('plugins:')
        for k, v in self.plugins.items():
            print(textwrap.indent(
                text='{} : \n{}'.format(
                    k, textwrap.indent(pprint.pformat(v), prefix=_INDENT.ttab)),
                prefix=_INDENT.ttab))

    def __repr__(self):
        return "Test({})".format(os.path.basename(self.test_dir))


def add_state(*test_dirs, state):
    res = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            vf = os.path.join(test_dir, state)
            if not os.path.exists(vf):
                with open(vf, 'w') as w:
                    w.write('1')
                res.append(test_dir)
    return res


def remove_state(*test_dirs, state):
    res = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            vf = os.path.join(test_dir, state)
            if os.path.exists(vf):
                os.remove(vf)
                res.append(test_dir)
    return res


def hide_tests(*test_dirs):
    return add_state(*test_dirs, state=_FNAME.hide)


def show_tests(*test_dirs):
    return remove_state(*test_dirs, state=_FNAME.hide)


def fav(*test_dirs):
    return add_state(*test_dirs, state=_FNAME.fav)


def unfav(*test_dirs):
    return remove_state(*test_dirs, state=_FNAME.fav)
