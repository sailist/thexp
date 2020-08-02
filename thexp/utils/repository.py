import json
import os
import sys
from functools import lru_cache
from typing import List
from uuid import uuid4

from git import Git, Repo

from thexp.utils.paths import renormpath
from .dates import curent_date
from ..globals import _GITKEY, _OSENVI, _FNAME

thexp_gitignores = ['.thexp/', _FNAME.repo, _FNAME.expsdirs, '.idea/']
py_gitignore = "\n".join(['# Byte-compiled / optimized / DLL files', '__pycache__/', '*.py[cod]',
                          '*$py.class', '', '# C extensions', '*.so', '', '# Distribution / packaging',
                          '.Python', 'build/', 'develop-eggs/', 'dist/', 'downloads/', 'eggs/', '.eggs/',
                          'lib/', 'lib64/', 'parts/', 'sdist/', 'var/', 'wheels/', 'pip-wheel-metadata/',
                          'share/python-wheels/', '*.egg-info/', '.installed.cfg', '*.egg', 'MANIFEST',
                          '', '# PyInstaller',
                          '#  Usually these files are written by a python script from a template',
                          '#  before PyInstaller builds the exe, so as to inject date/other infos into '
                          'it.',
                          '*.manifest', '*.spec', '', '# Installer logs', 'pip-log.txt',
                          'pip-delete-this-directory.txt', '', '# Unit test / coverage reports',
                          'htmlcov/', '.tox/', '.nox/', '.coverage', '.coverage.*', '.cache',
                          'nosetests.xml', 'coverage.xml', '*.cover', '.hypothesis/', '.pytest_cache/',
                          '', '# Translations', '*.mo', '*.pot', '', '# Django stuff:', '*.log',
                          'local_settings.py', 'db.sqlite3', '', '# Flask stuff:', 'instance/',
                          '.webassets-cache', '', '# Scrapy stuff:', '.scrapy', '',
                          '# Sphinx documentation', 'docs/_build/', '', '# PyBuilder', 'target/', '',
                          '# Jupyter Notebook', '.ipynb_checkpoints', '', '# IPython',
                          'profile_default/', 'ipython_config.py', '', '# pyenv', '.python-version', '',
                          '# celery beat schedule file', 'celerybeat-schedule', '',
                          '# SageMath parsed files', '*.sage.py', '', '# Environments', '.env', '.venv',
                          'env/', 'venv/', 'ENV/', 'env.bak/', 'venv.bak/', '',
                          '# Spyder project settings', '.spyderproject', '.spyproject', '',
                          '# Rope project settings', '.ropeproject', '', '# mkdocs documentation',
                          '/site', '', '# mypy', '.mypy_cache/', '.dmypy.json', 'dmypy.json', '',
                          '# Pyre type checker', '.pyre/'] + thexp_gitignores)


def _set_default_config(repo: Repo, names: List[str]):
    if len(names) == 0:
        return {}

    def _expsdir(repo):
        from .paths import global_config
        config = global_config()
        expsdir = config.get(_GITKEY.expsdir, renormpath(os.path.join(repo.working_dir, '.thexp/experiments')))
        return expsdir

    _default = {
        _GITKEY.uuid: lambda *args: uuid4().hex[:2],
        _GITKEY.expsdir: lambda repo: _expsdir(repo),
        _GITKEY.projname: lambda repo: renormpath(os.path.basename(repo.working_dir))
    }

    writer = repo.config_writer()
    res = {}
    for name in names:
        value = _default[name](repo)
        writer.add_value(_GITKEY.section_name, name, value)
        res[name] = value

    writer.write()
    writer.release()

    return res


def _check_section(repo: Repo):
    writer = repo.config_writer()
    if not writer.has_section(_GITKEY.section_name):
        writer.add_section(_GITKEY.section_name)
    writer.write()
    writer.release()


@lru_cache()
def git_config(repo: Repo):
    reader = repo.config_reader()
    if not reader.has_section(_GITKEY.section_name):
        _check_section(repo)
        reader = repo.config_reader()
    reader.read()
    try:
        config = {k: v for k, v in reader.items(_GITKEY.section_name)}
    except:
        config = {}

    lack_names = [i for i in {_GITKEY.expsdir, _GITKEY.uuid, _GITKEY.projname} if i not in config]
    _updates = _set_default_config(repo, lack_names)

    config.update(_updates)

    return config


def check_gitignore(repo: Repo, force=False):
    rp = os.path.join(repo.working_dir, _FNAME.expsdirs)
    if os.path.exists(rp) and not force:
        return

    ignorefn = os.path.join(repo.working_dir, _FNAME.gitignore)
    if not os.path.exists(ignorefn):
        with open(ignorefn, 'w', encoding='utf-8') as w:
            w.write(py_gitignore)
        return True
    else:
        amend = False
        with open(ignorefn, 'r', encoding='utf-8') as r:
            lines = [i.strip() for i in r.readlines()]
            for item in thexp_gitignores:
                if item not in lines:
                    lines.append(item)
                    amend = True
        if amend:
            with open(ignorefn, 'w', encoding='utf-8') as w:
                w.write('\n'.join(lines))
        return amend


def git_config_syntax(value: str):
    """
    git config 中对其中的一些字符需要转义，因为暂时只用到了路径的存储，因此这里只对路径进行转义
    :param value: git config 中的某个value
    :return:  转义后的 value
    """
    return value.replace('\\\\', '/').replace('\\', '/')


@lru_cache()
def git_root(dir="./"):
    """
    判断某目录是否在git repo 目录内（包括子目录），如果是，返回该 repo 的根目录
    :param dir:  要判断的目录。默认为程序运行目录
    :return: 如果是，返回该repo的根目录（包含 .git/ 的目录）
        否则，返回空
    """
    cur = os.getcwd()
    os.chdir(dir)
    try:
        res = Git().execute(['git', 'rev-parse', '--git-dir'])
    except Exception as e:
        print(e)
        res = None
    os.chdir(cur)
    return res


class branch:
    """
    用于配合上下文管理切换 git branch

    with branch(repo, branch):
        repo.index.commit('...')
    """

    def __init__(self, repo: Repo, branch: str):
        self.repo = repo
        self.old_branch = self.repo.head.reference
        self.branch = branch

    def __enter__(self):
        if self.branch not in self.repo.heads:
            head = self.repo.create_head(self.branch)
        else:
            head = self.repo.heads[self.branch]

        self.repo.head.reference = head
        # self.repo.head.reset(index=True, working_tree=True)
        return head

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.repo.head.reference = self.old_branch


def init_repo(dir='./'):
    """
    初始化某目录为 git repository，除了完成 git init 外，还会确认 thexp 需要的配置
    Args:
        dir:

    Returns:

    """
    path = git_root(dir)
    if path is not None:
        res = Repo(path)
    else:
        res = Repo.init(path)
    check_gitignore(repo=res, force=True)
    git_config(res)
    res.git.add('.')
    res.index.commit('initial commit')
    return res


@lru_cache()
def load_repo(dir='./') -> Repo:
    """
    尝试获取一个目录中的git repository 对象，如果不存在，那么会尝试建立后再获取；如果取消建立，则返回 None
    Args:
        dir: 目录
        如果 dir 为某 repo 的目录或子目录，那么将根据该目录的根目录返回 git Repo 对象
        如果 dir 为某新目录，那么将询问是否以该目录为根目录创建 Repo，若创建成功，则返回 Repo 对象，否则返回 None
        如果 dir = None，则以程序运行路径 (os.getcwd()) 或许程序 Repo

    Returns:
        Repo(...)，如果不存在或者取消创建，则返回 None
    """

    path = git_root(dir)

    if path is None:
        if _OSENVI.ignore_repo not in os.environ:
            print("fatal: not a git repository (or any of the parent directories)")
            print("-----------------------")
            path = input("type root path to init this project, \n(default: {}, type '!' to ignore".format(os.getcwd()))
        else:
            print("Variable 'repo' will be a None object. "
                  "Any operation that need this repo may cause Exception.")
            return None

        if '!' in path or '！' in path:
            print("Variable 'repo' will be a None object. "
                  "Any operation that need this repo may cause Exception.")
            return None

        res = Repo.init(path)
        check_gitignore(repo=res, force=True)
        # check_gitconfig(repo=res, force=True)
        res.git.add('.')
        res.index.commit('initial commit')
    else:
        res = Repo(path)
        amend = check_gitignore(repo=res, force=False)
        if amend:
            res.git.add(_FNAME.gitignore)
            res.index.commit('fix gitignore')
    return res


_commits_map = {}


def commit(repo: Repo, key=None, branch_name=_GITKEY.thexp_branch):
    """

    Args:
        key: 如果不为空，那么调用时会提交，否则不会重复提交相同key

    Returns:

    """
    if key is not None and key in _commits_map:
        return _commits_map[key]

    with branch(repo, branch_name):
        repo.git.add(all=True)
        commit_date = curent_date()
        commit_info = dict(
            date=commit_date,
            args=sys.argv,
            environ="jupyter" if "jupyter_core" in sys.modules else "python",
            version=sys.version,
        )
        commit_ = repo.index.commit(json.dumps(commit_info, indent=2))
    if key is not None:
        _commits_map[key] = commit_
    return commit_
