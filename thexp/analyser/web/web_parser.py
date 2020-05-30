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
import pandas as pd


def parse_tree(tree:dict):
    res = []
    for k,v in tree.items():
        ntree = {}
        ntree['label'] = k
        ntree['children'] = [{'parent':k,'label':vv} for vv in v['exps']]

        res.append(ntree)
    return res


def parse_base_table(dataframe:pd.DataFrame):
    """
            auto_train_loss.max	auto_train_loss.mean	auto_train_loss.min	auto_train_loss.std
    0001.5868fedf	4.618134498596191	4.142558177312215	3.7000625133514404	0.37553054908090666
    0002.5868fedf	2.3225433826446533	2.3119219144185386	2.3023736476898193	0.008269154456072377

    :param dataframe:
    :return:
    """
    table = []
    meta = []
    meta.append(dict(prop='test_name',label='试验ID'))
    for row in dataframe.iterrows():
        index,item = row
        for k in item.to_dict():
            meta.append(dict(prop=k,label=k))
        break

    for row in dataframe.iterrows():
        index,item = row
        res = item.to_dict()
        res['test_name'] = index
        table.append(res)
    return meta,table


def parse_stat_table(dataframe:pd.DataFrame):
    table = []
    meta = []
    meta.append(dict(prop='property',label='指标'))
    for row in dataframe.iterrows():
        index,item = row
        for k in item.to_dict():
            meta.append(dict(prop=k,label=k))
        break

    for row in dataframe.iterrows():
        index,item = row
        res = item.to_dict()
        res['property'] = index
        table.append(res)
    return meta,table


