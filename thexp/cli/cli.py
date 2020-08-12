import fire

doc = """
Usage:
 thexp init
 thexp board [--logdir=<logdir>]
 thexp board [--test=<test_name>] # find test_name and tensorboard it 
 thexp board  # default open ./board
 
 thexp reset <test_name>
 thexp reset --test=<test_name>
 thexp reset --test_name=<test_name>
 
 thexp archive <test_name>
 thexp archive --test=<test_name>
 thexp archive --test_name=<test_name>
"""
import sys
from thexp import __VERSION__
from thexp import Q

from thexp.decorators import regist_func

func_map = {}


@regist_func(func_map)
def init():
    import shutil
    import os

    templete_dir = os.path.join(os.path.dirname(__file__), 'templete')
    src_dir = os.getcwd()
    from thexp.utils.repository import init_repo

    dir_name = os.path.basename(src_dir)
    init_repo(src_dir)
    shutil.copytree(templete_dir, os.path.join(src_dir, dir_name))
    # os.rename(os.path.join(src_dir, 'templete'), os.path.join(src_dir, dir_name))
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py-tpl'):
                nfile = "{}.py".format(os.path.splitext(file)[0])
                os.rename(os.path.join(root, file), os.path.join(root, nfile))


@regist_func(func_map)
def check(*args, **kwargs):
    from thexp.utils.repository import init_repo
    init_repo()


def _board_with_logdir(logdir, *args, **kwargs):
    import os
    import subprocess

    tmpdir = os.path.join(os.path.dirname(logdir), 'board_tmp')
    os.makedirs(tmpdir, exist_ok=True)

    subprocess.check_call(['tensorboard', '--logdir={}'.format(logdir)],
                          env=dict(os.environ, TMPDIR=tmpdir))


def _board_with_test_name(test_name, *args, **kwargs):
    query = Q.tests(test_name)
    if query.empty:
        raise IndexError(test_name)
    vw = query.to_viewer()
    if not vw.has_board():
        raise AttributeError('{} has no board or has been deleted'.format(test_name))
    else:
        _board_with_logdir(vw.board_logdir, **kwargs)


@regist_func(func_map)
def board(*args, **kwargs):
    if len(args) > 0:
        kwargs.setdefault('test_name', args[0])

    if 'logdir' in kwargs:
        _board_with_logdir(**kwargs)
    elif 'test' in kwargs:
        _board_with_test_name(kwargs['test'])
    elif 'test_name' in kwargs:
        _board_with_test_name(**kwargs)
    else:
        _board_with_logdir('./board')


def _find_test_name(*args, **kwargs):
    if len(args) > 0:
        return args[0]
    elif 'test' in kwargs:
        return kwargs['test']
    elif 'test_name' in kwargs:
        return kwargs['test_name']
    return None


@regist_func(func_map)
def reset(*args, **kwargs):
    test_name = _find_test_name(*args, **kwargs)
    query = Q.tests(test_name)
    if query.empty:
        print("can't find test {}".format(test_name))
        exit(1)
    exp = query.to_viewer().reset()
    print('reset from {} to {}'.format(exp.plugins['reset']['from'], exp.plugins['reset']['to']))


@regist_func(func_map)
def archive(*args, **kwargs):
    test_name = _find_test_name(*args, **kwargs)
    query = Q.tests(test_name)
    if query.empty:
        print("can't find test {}".format(test_name))
        exit(1)
    exp = query.to_viewer().archive()

    print('archive {} to {}'.format(test_name, exp.plugins['archive']['file']))


def main(*args, **kwargs):
    if len(args) == 0 or 'help' in kwargs:
        print(doc)
        return

    branch = args[0]
    if branch in func_map:
        func_map[branch](*args[1:], **kwargs)
    else:
        print(doc)


fire.Fire(main)

exit(0)
