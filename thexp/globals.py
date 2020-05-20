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


class _REPOJ:
    repopath = 'repopath'
    exps = 'exps'


class _INFOJ:
    repo = 'repo'
    argv = 'argv'
    commit_hash = 'commit_hash'
    short_hash = 'short_hash'
    dirs = 'dirs'
    time_fmt = 'time_fmt'
    start_time = 'start_time'
    tags = 'tags'
    _tags = '_tags'
    _ = '_'
    plugins = 'plugins'
    end_time = 'end_time'
    end_code = 'end_code'


class _CONFIGL:
    user = 'user'
    exp = 'exp'
    repository = 'repository'


class _GITKEY:
    thexp = 'thexp'
    projname = 'projname'
    expsdir = 'expsdir'
    uuid = 'uuid'


class _FNAME:
    Exception = 'Exception'
    info = 'info.json'
    repo = 'repo.json'


class _ML:
    train = 'train'
    test = 'test'
    eval = 'eval'
    cuda = 'cuda'


class _BUILTIN_PLUGIN:
    trainer = 'trainer'
    writer = 'writer'
    logger = 'logger'
    saver = 'saver'
    rnd = 'rnd'

class _PLUGIN_WRITER:
    log_dir = 'log_dir'
    filename_suffix = 'filename_suffix'
    dir_name = 'board'


class _OSENVI:
    ignore_repo = 'ignore_repo'


class _INDENT:
    tab = '  '
    ttab = '    '
    tttab = '      '
