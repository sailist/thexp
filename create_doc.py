import inspect, thexp
from pprint import pprint
from thexp.base_classes import tree, attr
import os

cwd = os.getcwd()
mem = set()


def create_path_tree(path):
    res = attr()
    fs = os.listdir(path)
    for f in fs:
        af = os.path.join(path, f)
        if os.path.isdir(af):
            res.setdefault('childern', []).append(create_path_tree(af))
        else:
            res.setdefault('files', []).append(f)

    # if 'files' not in

    return res

#
# def finds(module):
#     dic = attr()
#
#     for root, dirs, fs in os.walk('thexp'):
#         for f in fs:
#             af = os.path.join(root, f)
#             if f == '__init__.py':
#                 inspect.getmoduleinfo(af)
#
#     return dic


val = finds(thexp)
