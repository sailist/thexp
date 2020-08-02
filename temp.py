import os
import subprocess
import sys

import fire

from thexp import Q
from thexp import __VERSION__
from thexp.decorators import regist_func
from thexp.globals import _PLUGIN_DIRNAME

doc = """
thexp {}
Usage:
 thexp init
 thexp board [--logdir=<logdir>] **kwargs
 thexp board [--test=<test_name>] **kwargs # find test_name and tensorboard it 
 thexp board # equal to tensorboard --logdir=./board
 
 thexp reset <test_name>
 thexp reset --test=<test_name>
 thexp reset --test_name=<test_name>
 
 thexp archive <test_name>
 thexp archive --test=<test_name>
 thexp archive --test_name=<test_name>
""".format(__VERSION__)

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
    shutil.copytree(templete_dir, src_dir)
    os.rename(os.path.join(src_dir, 'templete'), os.path.join(src_dir, dir_name))


def _board_with_logdir(logdir, tmpdir, *args, **kwargs):
    os.makedirs(tmpdir, exist_ok=True)
    os.environ['TMPDIR'] = tmpdir  # visible in this process + all children

    subprocess.Popen(
        ' '.join(['tensorboard', '--logdir={}'.format(logdir)] + ["--{}={}".format(k, v) for k, v in kwargs.items()]),
        env=dict(os.environ), preexec_fn=os.setsid,
        stdout=open('t.log', 'w', encoding='utf-8')
    )


def _board_with_test_name(test_name, *args, **kwargs):
    query = Q.tests(test_name)
    if query.empty:
        raise IndexError(test_name)
    vw = query.to_viewer()
    if not vw.has_board():
        raise AttributeError('{} has no board or has been deleted'.format(test_name))
    else:
        tmp_dir = os.path.join(vw.root, 'board_tmp')
        _board_with_logdir(vw.board_logdir, tmp_dir, **kwargs)


@regist_func(func_map)
def board(*args, **kwargs):
    if len(args) > 0:
        kwargs.setdefault('test_name', args[0])

    if 'logdir' in kwargs:
        tmpdir = os.path.join(os.path.dirname(kwargs['logdir']), _PLUGIN_DIRNAME.writer_tmp)
        _board_with_logdir(tmpdir=tmpdir, **kwargs)
    elif 'test' in kwargs:
        _board_with_test_name(kwargs['test'])
    elif 'test_name' in kwargs:
        _board_with_test_name(**kwargs)
    else:
        _board_with_logdir('./board', _PLUGIN_DIRNAME.writer_tmp, **kwargs)


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


stdout = sys.stdout


def main(*args, **kwargs):
    if len(args) == 0:
        return doc
    branch = args[0]
    if branch in func_map:
        func_map[branch](*args[1:], **kwargs)
    else:
        return doc


def simple_display(lines, out):
    text = '\n'.join(lines) + '\n'
    print(text, out=out)


fire.Fire(main)
# fire.__version__
