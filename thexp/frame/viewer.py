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
import copy
import json
import os

from ..globals import _INFOJ, _FNAME, _OSENVI, _REPOJ, _INDENT, _BUILTIN_PLUGIN

os.environ[_OSENVI.ignore_repo] = 'True'

from ..utils.generel_util import date_from_str, home_dir
import pprint


class SummaryViewer():
    def __init__(self):
        self._repos_info = None

    @property
    def repos_info(self):
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
        self._repos_info = None

    @property
    def repos(self):
        return list(self.repos_info.keys())

    @property
    def repopaths(self):
        return [i[_REPOJ.repopath] for i in self.repos_info.values()]

    def get_exps(self, repo, to_viewer=False):
        if repo in self.repos_info:
            if to_viewer:
                return [ExpViewer(i) for i in self.repos_info[repo][_REPOJ.exps].values() if os.path.exists(i)]
            return self.repos_info[repo][_REPOJ.exps]

        return None

    def get_repopath(self, repo):
        if repo in self.repos_info:
            return self.repos_info[repo][_REPOJ.repopath]

        return None


class ExpViewer():
    """
    :param trainer_cls: cls of trainer or str
    """

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self._exp_name = os.path.basename(exp_dir)

    @property
    def test_dirs(self):
        if not os.path.exists(self.exp_dir):
            return []
        fs = [os.path.join(self.exp_dir, f) for f in os.listdir(self.exp_dir) if not f.startswith('-')]
        return fs

    @property
    def hide_dirs(self):
        if not os.path.exists(self.exp_dir):
            return []
        fs = [os.path.join(self.exp_dir, f) for f in os.listdir(self.exp_dir) if f.startswith('-')]
        return fs

    @property
    def test_viewers(self):
        return [TestViewer(f) for f in self.test_dirs]

    def hide_tests(self, *test_dirs):
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                name = os.path.basename(test_dir)
                if name.startswith('-'):
                    return
                nname = '-{}'.format(name)
                os.rename(test_dir, os.path.join(self.exp_dir, nname))

    def show_tests(self, *test_dirs):
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                name = os.path.basename(test_dir)
                nname = name.lstrip('-')
                os.rename(test_dir, os.path.join(self.exp_dir, nname))


class TestViewer():
    def __init__(self, test_dir: str):
        self.test_dir = test_dir

        self.c, self.commithash = os.path.basename(test_dir).split(".")
        self._json_info = None

    @property
    def json_fn(self):
        return os.path.join(self.test_dir, _FNAME.info)

    @property
    def json_info(self):
        if self._json_info is not None:
            return self._json_info

        if not os.path.exists(self.json_fn):
            self._json_info = {}
        with open(self.json_fn, encoding='utf-8') as r:
            self._json_info = json.load(r)
        return self._json_info

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
    def board_reader(self):
        from .drawer import BoardReader
        from ..globals import _PLUGIN_WRITER

        writer = self.get_plugin(_BUILTIN_PLUGIN.writer)
        fs = os.listdir(writer[_PLUGIN_WRITER.log_dir])
        fs = [i for i in fs if i.endswith('.bd') and i.startswith('events.out.tfevents')]
        if len(fs) == 0:
            return None
        if len(fs) > 1:
            import warnings
            warnings.warn('found multi tfevents file, return the first one')
        file = os.path.join(writer[_PLUGIN_WRITER.log_dir],fs[0])

        return BoardReader(file)

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
    def plugins(self):
        dic = copy.deepcopy(self.json_info[_INFOJ.plugins])
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
        from .meter import Meter
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

    # TODO 用来解plugin
    def regist_deplug(self):
        pass
