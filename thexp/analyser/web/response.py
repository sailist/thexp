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
import re
from collections import defaultdict

from flask import jsonify

from thexp.analyser.charts import Parallel, Curve
from thexp.base_classes.tree import tree
from thexp.analyser import deplugs
from . import web_parser
from ..compare import Comparer
from ..expviewer import SummaryViewer, ProjViewer, TestViewer


class BaseResponses():
    """

    """

    def __init__(self):
        self.viewer = SummaryViewer()
        self.info = {}
        self.name_map = tree()
        self.initial()

    def refresh(self):
        self.viewer = SummaryViewer()

    def initial(self):
        self.add_name_map('summary', 'Summary')
        self.add_name_map('compare', 'Compare')
        self.add_name_map('plot', 'Plot')
        self.add_name_map('base', 'Base')
        self.info['stat'] = self.get_methods('stat')

    def get_methods(self, mkey):
        res = []
        for key in dir(self):
            if key.startswith('{}_'.format(mkey)):
                attr = getattr(self, key, None)
                if callable(attr):
                    name = re.sub("^{}_".format(mkey), "", key)
                    prop = self.name_map.get(name, name).title()
                    res.append({'name': name, 'prop': prop})
        return res

    def _base_analyse(self, data):
        """
        analyse data from frontend's requests
        :param data: request.json['data']
        :return:
            results: a list of testviewer objects
            compare: a Comparer object of these testviewers
        """

        def exp():
            return defaultdict(list)

        res = defaultdict(exp)
        for record in data:
            res[record['project_iname']][record['exp_name']].append(record['test_name'])

        results = self.viewer[res]
        compare = Comparer(*results)
        return results, compare

    def get_method(self, key, mkey):
        res = getattr(self, "{}_{}".format(key, mkey), None)
        return res

    def regist_tab(self, name):
        res = self.info.setdefault('plugins', dict())
        res['methods'] = self.get_methods(name)

    def add_name_map(self, name, value):
        self.name_map[name] = value

    def base_layout(self):
        return jsonify(code=200, data=self.info)

    def info_exps(self, data=None, *args, **kwargs):
        """
        exp 的信息，主页左侧的树的信息
        :param data:
        :param args:
        :param kwargs:
        :return:
        """
        tree = self.viewer.tree(1)
        res = web_parser.parse_tree(tree)
        return jsonify(data=res, code=200)

    def info_tests(self, exps=None, include_hide=False, *args, **kwargs):
        """
        test 的信息，主页的表格，每次左侧树的选中状态变更时进行调用并更新
        :param exps: request.json['data']
        :param all: 是否包括隐藏文件
        :param args:
        :param kwargs:
        :return:
        """
        res = defaultdict(list)
        for val in exps:
            res[val['project_iname']].append(val['exp_name'])
        proj_viewers = {i.proj_name: i for i in self.viewer.proj_viewers if i.proj_name in res}

        resp = []
        for proj_name, proj_viewer in proj_viewers.items():  # type:(str,ProjViewer)
            exp_names = res[proj_name]
            test_viewers = proj_viewer.filter(lambda x: x.exp_name in exp_names, include_hide=include_hide)

            resp.extend(test_viewers)
        resp = sorted(resp, key=lambda x: x.visible, reverse=True)

        return jsonify(code=200, data=[r.json_info for r in resp])

    def info_test(self, data=None, *args, **kwargs):
        """
        用于获取一个test的详细信息
        :param data: request.json['test_name']
        :param args:
        :param kwargs:
        :return:
        """
        tv = self.viewer.find(data)[0]  # type:TestViewer
        res = {}

        pops = {'plugins', 'tags', 'time_fmt', 'start_time', 'end_time'}

        tji = tv.json_info
        [tji.pop(i) for i in pops]
        res['base'] = tji

        ps = []
        ups = []
        for k, v in tv.plugins.items():
            if k in deplugs.deplug_map:
                res[k] = deplugs.deplug_map[k](tv, v)
                ps.append(k)
            else:
                ups.append(k)

        return jsonify(code=200, data=res, plugins=ps, unplug=ups)

    def git_reset(self, data=None, *args, **kwargs):
        """
        详见
        :param data:  request.json['test_name']
        :param args:
        :param kwargs:
        :return:
        """
        from thexp.utils.gitutils import reset
        res = self.viewer.find(data)[0]

        exp = reset(res)
        return jsonify(code=200)

    def git_archive(self, data=None, *args, **kwargs):
        """
        @see: thexp.utils.gitutils.archive
        :param data: request.json['test_name']
        :param args:
        :param kwargs:
        :return:
        """
        from thexp.utils.gitutils import archive
        viewer = SummaryViewer()
        res = viewer.find(data)[0]
        exp = archive(res)
        return jsonify(code=200, dir=exp.test_dir)


class Responses(BaseResponses):

    def stat_compare(self, data=None, *args, **kwargs):
        """
        用于绘制显示对比多个 test 的结果
        :param data: request.json['data']
        :return:
            JsonResponse
            {
                code=200,
                options=options
            }

        options 是Echarts的options
        """

        results, compare = self._base_analyse(data)
        pd, md = compare.paraller_dict()
        p = Parallel([i.test_name for i in results], pd, md, 'Compare')
        options = json.loads(p.echarts().dump_options_with_quotes())
        return jsonify(code=200, options=options)

    def stat_plot(self, data=None, *args, **kwargs):
        """
        用于绘制对比多个 test 的所有训练过程中记录参数的曲线
        :param data: request.json['data']
        :return:
            {
                code=200,
                data={
                    test_name:options
                }
            }
            options 是Echarts的options
        """
        results, compare = self._base_analyse(data)
        curves = compare.curves_dict()
        options = {k: json.loads(Curve(v, k).echarts().dump_options_with_quotes()) for k, v in curves.items()}
        return jsonify(dict(code=200, data=options))

    def stat_base(self, data=None, *args, **kwargs):
        """
        用于显示多个 test 的训练过程中各个记录参数的基本统计值
            第一列为试验ID，之后每一列分别为每一个记录参数的统计值
                包括 最大值、最小值、均值、方差
        :param data: request.json['data']
        :return:
            {
                code=200,
                meta=meta,
                table=table
            }
            meta 用于 vue-element中的表格的元信息生成，格式为一个list，每个元素为一个字典，
             [{label:<label-1>,prop:<prop-1>}, ... ]
                 <label> 表示表格相应列显示文字
                 <prop> 表示向表数据的哪个对应元素寻找
            table 表示基于元信息的数据，格式为一个list，每个元素为一个字典，表示一行数据，
            [{<prop-1>:<value>, <prop-2>:<value>...}, ... ]
        """
        results, compare = self._base_analyse(data)
        df = compare.statistics()
        meta, table = web_parser.parse_base_table(df)
        return jsonify(code=200, meta=meta, table=table)

    def stat_summary(self, data=None, *args, **kwargs):
        """
        用于显示多个 test 的训练过程中各个记录参数的基本统计值的统计值
            一共四行，
                分别为每一个记录参数的最大值最小值均值方差的 最大值、最小值、方差、均值
        :param data: request.json['data']
        :return:
            格式参考 stat_base() 方法
        """
        results, compare = self._base_analyse(data)
        df = compare.merge_statistics()
        meta, table = web_parser.parse_stat_table(df)
        return jsonify(code=200, meta=meta, table=table)
