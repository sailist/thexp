"""
这一部分定义了一些全局字符串，防止出现命名错误的情况
"""


class _REPOJ:
    repopath = 'repopath'
    exps = 'exps'
    exp_root = 'exp_root'


class _INFOJ:
    repo = 'repo'
    argv = 'argv'
    test_name = 'test_name'
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
    running = 'user'
    globals = 'globals'
    repository = 'repository'


class _GITKEY:
    thexp = 'thexp'
    projname = 'projname'
    expsdir = 'expsdir'
    uuid = 'uuid'
    section_name = 'thexp'
    thexp_branch = 'experiment'
    commit_key = 'thexp-commit'


class _FNAME:
    Exception = 'Exception'
    info = 'info.json'
    repo = 'repo.json'
    params = 'params.json'
    hide = '.hide'
    fav = '.fav'


class _ML:
    train = 'train'
    test = 'test'
    eval = 'eval'
    cuda = 'cuda'


class _BUILTIN_PLUGIN:
    trainer = 'trainer'
    params = 'params'
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


class _DLEVEL:
    proj = 'proj'
    exp = 'exp'
    test = 'test'