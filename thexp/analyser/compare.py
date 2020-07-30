"""
TODO 该类下各个功能还需要思考是否保留以及用途
"""

import pandas as pd

from thexp.analyser.viewer import TestViewer
from ..frame.params import BaseParams

_ignore_key = {'eidx', 'idx', 'device','global_step'}


class Metric:
    def __init__(self, key, type='max'):
        self.key = key
        self.type = type

    def from_dict(self, dic: dict):
        Metric(dic['name'], dic.get('type', 'max'))

    @staticmethod
    def guess(key:str, values):
        type = None
        if 'loss' in key.lower():
            type = 'min'
        elif 'acc' in key.lower() or 'accuracy' in key.lower():
            type = 'max'
        elif 'top' in key.lower():
            type = 'max'
        elif values[0] < values[-1]:
            type = 'max'
        else:
            type = 'min'

        return Metric(key, type)


class Singleton:
    def __init__(self,tv):
        self.tv = tv



class Comparer:
    def __init__(self, *tvs: TestViewer):
        self.tvs = tvs
        self._bds = None

    def summary(self):
        return pd.DataFrame([dict(i.json_info.walk()) for i in self.tvs])

    @property
    def boards(self):
        if self._bds is None:
            self._bds = [i.board_reader for i in self.tvs if i.board_reader is not None]
        return self._bds

    @property
    def params(self):
        return [tv.params for tv in self.tvs]

    def statistics(self):
        """统计传入bd的最大值、最小值、以及其他一些基本的数据"""
        from .summary import ValueSummary
        res = []
        for bd in self.boards:
            line = {}
            for stag in bd.scalars_tags:
                tattr = ValueSummary(stag, bd.get_scalars(stag).values).to_attr()
                line.update(tattr.walk())
            res.append(line)
        return pd.DataFrame(res, index=[i.name for i in self.tvs])

    def merge_statistics(self):
        """
        统计传入的bd的最大值、最小值、这些最大值、最小值的均值、方差
        :return:
        """
        re = self.statistics()
        return pd.DataFrame((re.max(), re.min(), re.mean(), re.std()), index=['max', 'min', 'mean', 'std']).T

    def curves_dict(self):
        res = {}
        for bd,tv in zip(self.boards,self.tvs):
            for stag in bd.scalars_tags:
                figure = res.setdefault(stag,{})
                scalars = bd.get_scalars(stag)
                figure[tv.name] = {
                    'name':tv.name,
                    'x':scalars.steps,
                    'y':scalars.values,
                }
        return res

    def paraller_dict(self, params: BaseParams = None):
        """
        根据传入的Metric，构造绘制平行图的字典
        :param params: 约束，如果为空，则全都记录
        :return:
        """
        params_dicts = []
        metric_dicts = []

        _dic = {}
        for tv in self.tvs:
            for stag in tv.board_reader.scalars_tags:
                _dic[stag] = Metric.guess(stag, tv.board_reader.get_scalars(stag).values)

        metrics = _dic.values()

        for tv in self.tvs:
            params_dict = {}
            metric_dict = {}
            p = tv.params
            for k, v in p.inner_dict.walk():
                if params is None or k in params:
                    if k not in _ignore_key:
                        params_dict[k] = v

            for metric in metrics:
                if metric.key in tv.metrics:
                    metric_dict[metric.key] = tv.metrics[metric.key]
                else:
                    metric_dict[metric.key] = None

            params_dicts.append(params_dict)
            metric_dicts.append(metric_dict)

        return params_dicts, metric_dicts
