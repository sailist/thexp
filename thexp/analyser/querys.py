"""

viewer = GlobalViewer()

viewer.repos()['SemiEnhance'].exps()['some_exp'].tests()[0,1,2]...
viewer.repos()['SemiEnhance'].exps()['some_exp'].tests('0001','0002')...
viewer.exps()[:2].tests()
viewer.tests().first() -> TestViewer
viewer.tests().last() -> TestViewer
viewer.tests().last() -> TestViewer

    ... list all repos



"""
import json
import os
from itertools import chain
from pprint import pformat
from typing import List, Iterator
from datetime import datetime, timedelta
import pandas as pd

from thexp.base_classes.list import llist
from thexp.globals import FNAME_
from thexp.utils.paths import home_dir
from .reader import BoardReader
from ..utils.iters import is_same_type
from ..globals import BUILTIN_PLUGIN_

# 虽然好像没什么用但还是设置上了
pd.set_option('display.max_colwidth', 160)
pd.set_option('colheader_justify', 'center')


class Query:

    def __init__(self):
        pass

    def tests(self, *items):
        return self.repos().exps().tests(*items)

    def exps(self, *items):
        return self.repos().exps(*items)

    def repos(self, *items):
        if len(items) == 0:
            return ReposQuery(*self.projs_list)
        else:
            return ReposQuery(*self.projs_list)[items]

    @property
    def projs_list(self):
        global_repofn = os.path.join(home_dir(), FNAME_.repo)
        if not os.path.exists(global_repofn):
            res = {}
        else:
            with open(global_repofn, 'r', encoding='utf-8') as r:
                res = json.load(r)

        repos = []
        projs = []
        for k, v in res.items():  # log_dir(proj level), repopath
            repos.append(v)
            projs.append(k)
        return projs, repos


class ReposQuery:
    """
    以项目为单位的Query，每个项目可能对应多个实验存储路径（一般不主动更改的情况下只有一个）
    """

    def __init__(self, projs: List[str], repos: List[str]):
        self.projs = llist(projs)
        self.repos = llist(repos)

        self.proj_names = [os.path.basename(i) for i in self.projs]

    def __and__(self, other):
        if isinstance(other, ReposQuery):
            combine = set(self.projs) & set(other.projs)
            projs = []
            repos = []
            for proj, repo in chain(zip(self.projs, self.repos), zip(other.projs, other.repos)):
                if proj in combine:
                    combine.remove(proj)
                    projs.append(proj)
                    repos.append(repo)
            return ReposQuery(projs, repos)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, ReposQuery):
            combine = set(self.projs) | set(other.projs)
            projs = []
            repos = []
            for proj, repo in chain(zip(self.projs, self.repos), zip(other.projs, other.repos)):
                if proj in combine:
                    combine.remove(proj)
                    projs.append(proj)
                    repos.append(repo)
            return ReposQuery(projs, repos)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            assert is_same_type(items)
            for item in items:
                if isinstance(item, str):
                    try:
                        idx = self.proj_names.index(item)
                        res.append(idx)
                    except:
                        raise IndexError(item)
                elif isinstance(item, int):
                    res.append(item)
            return ReposQuery(self.projs[res], self.repos[res])  # do not have type error
        elif isinstance(items, int):
            res.append(items)
            return ReposQuery(self.projs[res], self.repos[res])  # do not have type error
        elif isinstance(items, str):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return ReposQuery(self.projs[items], self.repos[items])  # do not have type error

    def __repr__(self):
        return self.df().__repr__()

    __str__ = __repr__

    def _repr_html_(self):
        with pd.option_context('display.max_colwidth', 0):
            with pd.option_context('colheader_justify', 'center'):
                return self.df()._repr_html_()

    def df(self):
        res = []
        for repo, proj in zip(self.repos, self.projs):
            res.append((os.path.basename(proj), repo))
        return pd.DataFrame(res, columns=['name', 'Repo path'])

    def tests(self):
        return self.exps().tests()

    def exps(self, *items):
        if len(items) == 0:
            viewers = self.to_viewers()
            exp_dirs = []
            for viewer in viewers:
                exp_dirs.extend(viewer.exp_dirs)
            return ExpsQuery(exp_dirs)
        else:
            return self[items].exps()

    @property
    def empty(self):
        return len(self.repos) == 0

    @property
    def isitem(self):
        return len(self.repos) == 1

    def to_viewer(self):
        if self.isitem:
            from .viewer import ProjViewer
            return ProjViewer(self.projs[0])
        raise ValueError('only one projs contains can be converted to ProjViewer')

    def to_viewers(self):
        from .viewer import ProjViewer
        return [ProjViewer(i) for i in self.projs]


class ExpsQuery:
    def __init__(self, exp_dirs):
        self.exp_dirs = llist(exp_dirs)
        self.exp_names = [os.path.basename(i) for i in self.exp_dirs]

    def __and__(self, other):
        if isinstance(other, ExpsQuery):
            combine = set(self.exp_dirs) & set(other.exp_dirs)
            exp_dirs = []
            for exp_dir in chain(self.exp_dirs, other.exp_dirs):
                if exp_dir in combine:
                    combine.remove(exp_dir)
                    exp_dirs.append(exp_dir)
            return ExpsQuery(exp_dirs)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, ExpsQuery):
            combine = set(self.exp_dirs) | set(other.exp_dirs)
            exp_dirs = []
            for exp_dir in chain(self.exp_dirs, other.exp_dirs):
                if exp_dir in combine:
                    combine.remove(exp_dir)
                    exp_dirs.append(exp_dir)
            return ExpsQuery(exp_dirs)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            assert is_same_type(items)
            for item in items:
                if isinstance(item, str):
                    try:
                        idx = self.exp_names.index(item)
                        res.append(idx)
                    except:
                        raise IndexError(item)
                elif isinstance(item, int):
                    res.append(item)
            return ExpsQuery(self.exp_dirs[res])  # do not have type error
        elif isinstance(items, int):
            res.append(items)
            return ExpsQuery(self.exp_dirs[res])  # do not have type error
        elif isinstance(items, str):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return ExpsQuery(self.exp_dirs[items])  # do not have type error
        else:
            raise TypeError(items)

    def __repr__(self):
        if self.empty:
            return '[Empty]'
        return self.df().__repr__()

    __str__ = __repr__

    def _repr_html_(self):
        if self.empty:
            return '<pre>[Empty]</pre>'
        return self.df()._repr_html_()

    def df(self):
        res = []
        names = []
        for viewer in self.to_viewers():
            res.append(viewer.test_names)
            names.append(viewer.name)
        df = pd.DataFrame(res, index=names).T
        return df

    def tests(self, *items):
        if len(items) == 0:
            viewers = self.to_viewers()
            exp_dirs = []
            for viewer in viewers:
                exp_dirs.extend(viewer.test_dirs)
            return TestsQuery(exp_dirs)
        else:
            return self.tests()[items]

    @property
    def empty(self):
        return len(self.exp_dirs) == 0

    @property
    def isitem(self):
        return len(self.exp_dirs) == 1

    def to_viewer(self):
        if self.isitem:
            from .viewer import ExpViewer
            return ExpViewer(self.exp_dirs[0])
        raise ValueError('only one projs contains can be converted to ProjViewer')

    def to_viewers(self):
        from .viewer import ExpViewer
        return [ExpViewer(i) for i in self.exp_dirs]


class BoardQuery():
    def __init__(self, board_readers: List[BoardReader], test_names: List[str]):
        self.board_readers = llist(board_readers)
        self.test_names = llist(test_names)

    def __and__(self, other):
        if isinstance(other, BoardQuery):
            combine = set(self.test_names) & set(other.test_names)
            readers = []
            test_names = []
            for reader, test_name in chain(zip(self.board_readers, self.test_names),
                                           zip(other.board_readers, other.test_names)):
                if test_name in combine:
                    combine.remove(test_name)
                    readers.append(reader)
                    test_names.append(test_name)
            return BoardQuery(readers, test_names)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, BoardQuery):
            combine = set(self.test_names) | set(other.test_names)
            readers = []
            test_names = []
            for reader, test_name in chain(zip(self.board_readers, self.test_names),
                                           zip(other.board_readers, other.test_names)):
                if test_name in combine:
                    combine.remove(test_name)
                    readers.append(reader)
                    test_names.append(test_name)
            return BoardQuery(readers, test_names)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            assert is_same_type(items, int)
            for item in items:
                if isinstance(item, int):
                    res.append(item)
            return BoardQuery(self.board_readers[res], self.test_names[res])  # do not have type error
        elif isinstance(items, int):
            res.append(items)
            return BoardQuery(self.board_readers[res], self.test_names[res])  # do not have type error
        elif isinstance(items, str):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return BoardQuery(self.board_readers[items], self.test_names[items])  # do not have type error

    def __repr__(self):
        return pformat(self.test_names)

    __str__ = __repr__

    @property
    def scalar_tags(self):
        """返回所有board reader 重合的tags"""
        from functools import reduce
        from operator import and_

        scalars_tags = [set(i.scalars_tags) for i in self.board_readers]
        return reduce(and_, set(scalars_tags))

    def has_scalar_tags(self, tag):
        res = []
        for i, reader in enumerate(self.board_readers):
            if tag in reader.scalars_tags:
                res.append(i)
        return self[res]

    def values(self, tag, with_step=False):
        res = []
        for reader in self.board_readers:
            try:
                val = reader.get_scalars(tag)
                if not with_step:
                    res.append(val.value)
            except:
                res.append(None)
        return res

    def curve(self, tag, backend='matplotlib'):
        from .charts import Curve
        figure = {}
        for bd, test_name in zip(self.board_readers, self.test_names):
            scalars = bd.get_scalars(tag)
            figure[test_name] = {
                'name': test_name,
                'x': scalars.steps,
                'y': scalars.values,
            }
        curve = Curve(figure)
        plot_func = getattr(curve, backend, None)
        if plot_func is None:
            raise NotImplementedError(backend)
        else:
            return plot_func()


class TestsQuery:

    def __init__(self, test_dirs: List[str]):
        self.test_dirs = llist(test_dirs)
        self.test_names = [os.path.basename(i) for i in self.test_dirs]

    def __and__(self, other):
        if isinstance(other, TestsQuery):
            combine = set(self.test_names) & set(other.test_names)
            test_dirs = []
            for test_dir, test_name in chain(zip(self.test_dirs, self.test_names),
                                             zip(other.test_dirs, other.test_names)):
                if test_name in combine:
                    combine.remove(test_name)
                    test_dirs.append(test_dir)
            return TestsQuery(test_dirs)
        raise TypeError(self, other)

    def __or__(self, other):
        if isinstance(other, TestsQuery):
            combine = set(self.test_names) | set(other.test_names)
            test_dirs = []
            for test_dir, test_name in chain(zip(self.test_dirs, self.test_names),
                                             zip(other.test_dirs, other.test_names)):
                if test_name in combine:
                    combine.remove(test_name)
                    test_dirs.append(test_dir)
            return TestsQuery(test_dirs)
        raise TypeError(self, other)

    def __getitem__(self, items):
        res = []
        if isinstance(items, (Iterator, list, tuple)):
            assert is_same_type(items)
            for item in items:
                if isinstance(item, str):
                    try:
                        idx = self.test_names.index(item)
                        res.append(idx)
                    except:
                        raise IndexError(item)
                elif isinstance(item, int):
                    res.append(item)
            return TestsQuery(self.test_dirs[res])  # do not have type error
        elif isinstance(items, int):
            res.append(items)
            return TestsQuery(self.test_dirs[res])  # do not have type error
        elif isinstance(items, str):
            return self.__getitem__([items])
        elif isinstance(items, slice):
            return TestsQuery(self.test_dirs[items])  # do not have type error

    def __repr__(self):
        if self.empty:
            return '[Empty]'
        return self.df().__repr__()

    def _repr_html_(self):
        if self.empty:
            return '<pre>[Empty]</pre>'
        return self.df()._repr_html_()

    __str__ = __repr__

    def df(self):
        df = pd.DataFrame([viewer.df_info() for viewer in self.to_viewers()],
                          columns=self[0].to_viewer().df_columns(),
                          index=self.test_names)
        return df

    def boards(self):
        return BoardQuery([i.board_reader for i in self.to_viewers()], self.test_names)

    @property
    def empty(self):
        return len(self.test_dirs) == 0

    @property
    def isitem(self):
        return len(self.test_dirs) == 1

    def to_viewers(self):
        from .viewer import TestViewer
        return [TestViewer(i) for i in self.test_dirs]

    def to_viewer(self):
        if self.isitem:
            from .viewer import TestViewer
            return TestViewer(self.test_dirs[0])
        raise ValueError('only one test_dirs contains can be converted to TestViewer')

    def first(self):
        return self[0]

    def last(self):
        return self[-1]

    def range(self, left_time=None, right_time=None):
        """
        筛选 start_time 位于某时间区间内的 test
        Args:
            left_time:
            right_time:

        Returns:

        """
        if left_time is None and right_time is None:
            return self

        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if left_time is None:
                if viewer.start_time < right_time:
                    res.append(i)
            elif right_time is None:
                if viewer.start_time > left_time:
                    res.append(i)
            else:
                if viewer.start_time > left_time and viewer.start_time < left_time:
                    res.append(i)

        return self[res]

    def success(self):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.success_exit:
                res.append(i)
        return self[res]

    def failed(self):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if not viewer.success_exit:
                res.append(i)
        return self[res]

    def has_tag(self, tag: str, toggle=True):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.has_tag(tag):
                if toggle:
                    res.append(i)
            else:
                if not toggle:
                    res.append(i)
        return self[res]

    def has_plugin(self, plugin, toggle=True):
        res = []
        for i, viewer in enumerate(self.to_viewers()):
            if viewer.has_plugin(plugin):
                if toggle:
                    res.append(i)
            else:
                if not toggle:
                    res.append(i)

        return self[res]

    def has_board(self, toggle=True):
        return self.has_plugin(BUILTIN_PLUGIN_.writer, toggle)

    def has_log(self, toggle=True):
        return self.has_plugin(BUILTIN_PLUGIN_.logger, toggle)

    def has_modules(self, toggle=True):
        return self.has_plugin(BUILTIN_PLUGIN_.saver, toggle)

    def has_params(self, toggle=True):
        return self.has_plugin(BUILTIN_PLUGIN_.params, toggle)

    def has_trainer(self, toggle=True):
        return self.has_plugin(BUILTIN_PLUGIN_.trainer, toggle)

    def in_time(self,
                minutes=0,
                hours=0,
                days=0,
                weeks=0,
                seconds=0):
        delta = timedelta(minutes=minutes, days=days, hours=hours, weeks=weeks, seconds=seconds)
        return self.range(left_time=datetime.now() - delta)

    def delete(self):
        for viewer in self.to_viewers():
            viewer.delete()

    def delete_board(self):
        for viewer in self.has_board(True).to_viewers():
            viewer.delete_board()

    def delete_log(self):
        for viewer in self.has_log(True).to_viewers():
            viewer.delete_log()

    def delete_modules(self):
        pass  # TODO

    def delete_keypoints(self):
        pass  # TODO

    def delete_checkpoints(self):
        pass  # TODO

    def delete_models(self):
        pass  # TODO


Q = Query()

