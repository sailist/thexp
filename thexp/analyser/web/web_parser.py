"""

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


