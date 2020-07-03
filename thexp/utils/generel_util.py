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
import hashlib
import os
import sys

from datetime import datetime

from ..globals import _GITKEY


def listdir_by_time(dir_path):
    dir_list = os.listdir(dir_path)
    if not dir_list:
        return []
    else:
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list, key=lambda x: os.path.getatime(os.path.join(dir_path, x)), reverse=True)
        return dir_list


def _default_config():
    return {
        'expsdir': os.path.expanduser("~/.thexp/experiments")
    }


def _create_home_dir(path):
    os.makedirs(path)
    import json
    with open("config.json", "w") as w:
        json.dump({
            _GITKEY.expsdir: os.path.join(path, 'experiments')
        }, w, indent=2)


def home_dir():
    path = os.path.expanduser("~/.thexp")
    if not os.path.exists(path):
        _create_home_dir(path)

    return path


def config_path():
    return os.path.join(home_dir(), "config.json")


def file_atime_hash(file):
    return string_hash(str(os.path.getatime(file)))


def string_hash(*str):
    hl = hashlib.md5()
    for s in str:
        hl.update(s.encode(encoding='utf-8'))
    return hl.hexdigest()[:16]


def file_hash(file):
    hl = hashlib.md5()
    with open(file, encoding="utf-8") as r:
        s = "".join(r.readlines())
        hl.update(s.encode(encoding='utf-8'))
    return hl.hexdigest()[:16]


def curent_date(fmt='%y-%m-%d-%H%M%S', dateobj: datetime = None):
    if dateobj is not None:
        return dateobj.strftime(fmt)
    return datetime.now().strftime(fmt)


def date_from_str(value, fmt='%y-%m-%d-%H%M%S') -> datetime:
    return datetime.strptime(value, fmt)


def file_atime2date(file, fmt='%y%m%d-%H%M%S') -> str:
    """获取某文件的 atime """
    return curent_date(fmt, datetime.fromtimestamp(os.path.getatime(file)))


def path_equal(p1: str, p2: str) -> bool:
    """
    判断两个路径是否相同  TODO（疑似在linux下不好用）
    Args:
        p1:
        p2:

    Returns:

    """
    return os.path.normcase(p1) == os.path.normcase(p2)


def path_in(sub: str, all: str) -> bool:
    """
    判断某路径是另一路径的子路径
    Args:
        sub:
        all:

    Returns:

    """
    return os.path.normcase(sub) in os.path.normcase(all)


def filter_filename(title, substr='-'):
    """
    过滤非法字符
    Args:
        title:
        substr:

    Returns:

    """
    import re
    title = re.sub('[\/:*?"<>|]', substr, title)  # 去掉非法字符
    return title


def iter2pair(obj):
    """"""
    for k in obj:
        if isinstance(obj, dict):
            yield k, obj[k]
        elif isinstance(k, (list, tuple)):
            yield k[0], k[1]
        elif isinstance(k, dict):
            for kk, vv in k.items():
                yield kk, vv


def hash(value) -> str:
    import hashlib
    from collections.abc import Iterable
    from numbers import Number
    from numpy import ndarray
    from torch import Tensor
    hl = hashlib.md5()

    if isinstance(value, (ndarray, Tensor)):
        if isinstance(hl, Tensor):
            value = value.detach_().cpu().numpy()
        try:
            value = value.item()
        except:
            value = None
    if isinstance(value, (Number)):
        value = str(value)

    if isinstance(value, dict):
        for k in sorted(value.keys()):
            v = value[k]
            hl.update(str(k).encode(encoding='utf-8'))
            hl.update(hash(v).encode(encoding='utf-8'))
    elif isinstance(value, str):
        hl.update(value.encode(encoding='utf-8'))
    elif isinstance(value, Iterable):
        for v in value:
            hl.update(hash(v).encode(encoding='utf-8'))
    return hl.hexdigest()


def deep_chain(item):
    from collections.abc import Iterable
    if isinstance(item, Iterable):
        for i in item:
            if isinstance(i, Iterable):
                for ii in deep_chain(i):
                    yield ii
            else:
                yield i
    else:
        yield item


def renormpath(path):
    return os.path.normcase(path).replace("\\", '/')
