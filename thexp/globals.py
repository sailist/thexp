"""
这一部分定义了一些全局字符串，防止出现命名错误的情况
"""


class REPOJ_:
    repopath = 'repopath'
    exps = 'exps'
    exp_root = 'exp_root'


class INFOJ_:
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


class CONFIGL_:
    running = 'user'
    globals = 'globals'
    repository = 'repository'


class GITKEY_:
    thexp = 'thexp'
    projname = 'projname'
    expsdir = 'expsdir'
    uuid = 'uuid'
    section_name = 'thexp'
    thexp_branch = 'experiment'
    commit_key = 'thexp-commit'


class FNAME_:
    Exception = 'Exception'
    info = 'info.json'
    repo = 'repo.json'
    params = 'params.json'
    repopath = '.repopath'
    expsdirs = '.expsdirs'
    gitignore = ".gitignore"


class TEST_BUILTIN_STATE_:
    hide = 'hide'
    fav = 'fav'


class ML_:
    train = 'train'
    test = 'test'
    eval = 'eval'
    cuda = 'cuda'


class BUILTIN_PLUGIN_:
    trainer = 'trainer'
    params = 'params'
    writer = 'writer'
    logger = 'logger'
    saver = 'saver'
    rnd = 'rnd'


class PLUGIN_DIRNAME_:
    writer = 'board'
    writer_tmp = 'board_tmp'
    saver = 'modules'
    rnd = 'rnd'


class PLUGIN_WRITER_:
    log_dir = 'log_dir'
    filename_suffix = 'filename_suffix'
    dir_name = 'board'


class OSENVI_:
    ignore_repo = 'ignore_repo'


class INDENT_:
    tab = '  '
    ttab = '    '
    tttab = '      '


class DLEVEL_:
    proj = 'proj'
    exp = 'exp'
    test = 'test'


class OS_ENV_:
    CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'
