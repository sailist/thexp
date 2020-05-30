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
import json
import os
import sys
from uuid import uuid4

from git import Git, Commit, Repo
from gitdb.util import hex_to_bin

from thexp.utils.generel_util import renormpath
from .generel_util import curent_date
from ..analyser.expviewer import TestViewer
from ..frame.experiment import Experiment
from ..globals import _GITKEY, _OSENVI

hgit = Git()

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
                          '# Pyre type checker', '.pyre/', '', '.thexp/', 'repo.json'])


def check_gitignore(dir):
    ignorefn = os.path.join(dir, ".gitignore")
    if not os.path.exists(ignorefn):
        with open(ignorefn, 'w', encoding='utf-8') as w:
            w.write(py_gitignore)
    else:
        with open(ignorefn, 'r', encoding='utf-8') as r:
            lines = [i.strip() for i in r.readlines()]
            if '.thexp/' not in lines:
                lines.append('.thexp/')
            if 'repo.json' not in lines:
                lines.append('repo.json')

        with open(ignorefn, 'w', encoding='utf-8') as w:
            w.write('\n'.join(lines))
        # if not have_proj:
        #     with open(ignorefn, 'a', encoding='utf-8') as w:
        #         w.write("\r\n.thexp/\r\n")
        # if not have_repofn:
        #     with open(ignorefn, 'a', encoding='utf-8') as w:
        #         w.write("\r\nrepo.json\r\n")


class ExpRepo:
    """
    和试验相对应的 Repo 对象，包含该 repo 的唯一命名，
    """
    _instance = None

    def __init__(self):
        self.repo = load_repo()
        self._check_git_config()
        self.commit = None
        # self.check_commit()

    def check_commit(self):
        """
        检查当前对象是否提交
        :return:
        """
        with branch(self.repo, _GITKEY.thexp_branch):
            self._check_commit_exp()

    def _check_git_config(self):
        """
        该方法检查与git相关的两项内容
        1. 检查 thexp 为每个repo添加的配置项，会包括以下内容（存放在 [thexp] 中）：
            uuid：每个git repo会生成一个两位id，用于避免不同的项目的项目名出现重复（重复概率 1/1296）
            projname：当前的项目名，默认为 repo 根目录的目录文件名
            expsdir：该 repo 中所有实验的存储路径，默认为该 repo 目录下的 .thexp/experiments/
        2. 检查 .gitignore，若该文件存在，则为其中添加 .thexp/ 和 repo.json 两项（如果没有）；
            否则，为该 repo 生成 .gitignore 文件
            该文件内容参考 https://github.com/github/gitignore/edit/master/Python.gitignore，并在此基础上
            添加了 .thexp/ 和 repo.json 两项
        :return:
        """
        writer = self.repo.config_writer()
        if not writer.has_section(_GITKEY.section_name):
            writer.add_section(_GITKEY.section_name)

        expsdir = writer.get_value(_GITKEY.section_name, _GITKEY.expsdir, "")
        if expsdir == '':
            expsdir = renormpath(os.path.join(self.repo.working_dir, '.thexp/experiments'))
            writer.add_value(_GITKEY.section_name, _GITKEY.expsdir,
                             expsdir)

        projname = writer.get_value(_GITKEY.section_name, _GITKEY.projname, "")
        if projname == '':
            projname = renormpath(os.path.basename(self.repo.working_dir))
            writer.add_value(_GITKEY.section_name, _GITKEY.projname, projname)

        uuid = writer.get_value(_GITKEY.section_name, _GITKEY.uuid, "")
        if uuid == '':
            uuid = uuid4().hex[:2]
            writer.add_value(_GITKEY.section_name, _GITKEY.uuid, uuid)
        elif isinstance(uuid, int):
            uuid = "{:02d}".format(uuid)

        projkey = '{}.{}'.format(projname, uuid)

        writer.write()
        writer.release()
        self.uuid, self.projkey = uuid, projkey
        self.expsdir, self.projname = expsdir, projname
        regist_repo(self)

    def _check_commit_exp(self):
        """
        对当前的文件状态进行一次提交，所有的提交均在 experiment 分支上完成，且在运行完后恢复到原分支，用于保证不污染原分支
        :return:
        """
        if ExpRepo._instance is None or ExpRepo._instance.commit is None:
            self.repo.git.add(all=True)
            self.commit_info = dict(
                date=curent_date(),
                args=sys.argv,
                environ="jupyter" if "jupyter_core" in sys.modules else "python",
                version=sys.version,
            )
            self.commit = self.repo.index.commit(json.dumps(self.commit_info, indent=2))
            self.commit_time = curent_date()

    @staticmethod
    def singleton():
        """
        获取单例模式，在任何时间使用该方法获取该类实例后，在运行期间只会进行一次提交
        否则，通过正常构造方法，每次获取一次该类实例均会创建一次提交。
        注意：在调用该方法创建实例后，即使通过正常构造方法创建该类实例，也不会进行提交，
            此时如果要完成提交，可以直接通过 gitPython 库中的命令完成提交，或运行 ExpRepo.clear_instance() 清除单例的实例
        :return:
        """
        if ExpRepo._instance is None:
            ExpRepo._instance = ExpRepo()

        return ExpRepo._instance

    @staticmethod
    def clear_instance():
        """清除ExpRepo实例，主要用于使新创建实例可提交"""
        ExpRepo._instance = None


def git_config_syntax(value: str):
    """
    git config 中对其中的一些字符需要转义，因为暂时只用到了路径的存储，因此这里只对路径进行转义
    :param value: git config 中的某个value
    :return:  转义后的 value
    """
    return value.replace('\\\\', '/').replace('\\', '/')


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
        # self.repo.head.reset(index=True, working_tree=True)


def load_repo(dir=None) -> Repo:
    """
    尝试获取一个目录中的git repository 对象
    :param dir: 目录
        如果 dir 为某 repo 的目录或子目录，那么将根据该目录的根目录返回 git Repo 对象
        如果 dir 为某新目录，那么将询问是否以该目录为根目录创建 Repo，若创建成功，则返回 Repo 对象，否则返回 None
        如果 dir = None，则以程序运行路径 (os.getcwd()) 或许程序 Repo
    :return: Repo(...)，如果不存在或者取消创建，则返回 None
    """
    if dir is None:
        path = git_root()
    else:
        path = git_root(dir)

    if dir is None:
        dir = os.getcwd()

    if path is None:
        if _OSENVI.ignore_repo not in os.environ:
            print("fatal: not a git repository (or any of the parent directories)")
            print("-----------------------")
            path = input("type root path to init this project, \n(default: {}, type '!' to ignore".format(dir))
        else:
            print("Variable 'repo' will be a None object. "
                  "Any operation that need this repo may cause Exception.")
            return None

        if '!' in path:
            print("Variable 'repo' will be a None object. "
                  "Any operation that need this repo may cause Exception.")
            return None

        check_gitignore(path)
        res = Repo.init(path)
        res.git.add(".")
        res.index.commit("initial commit")
    else:
        check_gitignore(path)
        res = Repo(path)
    return res


def regist_repo(exp_repo: ExpRepo):
    """
    在 repo 的根目录下注册实验名，用于 ProjViewer 检索
    :param exp_repo:  Exprepo对象
    :return:
    """
    from .generel_util import home_dir
    from ..globals import _FNAME
    projkey = exp_repo.projkey

    rp = os.path.join(home_dir(), _FNAME.repo)
    if not os.path.exists(rp):
        res = {}
    else:
        with open(rp, encoding='utf-8') as r:
            res = json.load(r)

    lis = res.setdefault(projkey, dict())

    lis['repopath'] = exp_repo.repo.working_dir
    lis['exp_root'] = exp_repo.expsdir

    with open(rp, 'w', encoding='utf-8') as w:
        json.dump(res, w, indent=2)

    rp = os.path.join(exp_repo.repo.working_dir, _FNAME.repo)
    if not os.path.exists(rp):
        res = {}
    else:
        with open(rp, encoding='utf-8') as r:
            res = json.load(r)
        if len(res.keys()) > 0 and projkey not in res:
            val = list(res.values())[0]
            res = {projkey: val}

    lis = res.setdefault(projkey, dict())
    lis['repopath'] = exp_repo.repo.working_dir
    lis['exp_root'] = exp_repo.expsdir
    with open(rp, 'w', encoding='utf-8') as w:
        json.dump(res, w, indent=2)


def regist_exps(exp_repo: ExpRepo, exp_name, path):
    """
    在 repo 目录下注册某实验 exp ，用于 ProjViewer 的检索
    :param exp_repo:
    :param exp_name: 该 exp 名
    :param path: 该 exp 的所有 test 存储根目录
    :return: 返回是否修改，如有修改，可能需要重新提交保证最新
    """
    from ..globals import _FNAME
    repo = exp_repo.repo
    projkey = exp_repo.projkey

    rp = os.path.join(repo.working_dir, _FNAME.repo)
    if not os.path.exists(rp):
        res = {}
    else:
        with open(rp, encoding='utf-8') as r:
            res = json.load(r)

    amend = False

    lis = res[projkey].setdefault('exps', [])
    if exp_name not in lis:
        lis.append(exp_name)
        amend = True

    with open(rp, 'w', encoding='utf-8') as w:
        json.dump(res, w, indent=2)

    return amend


def archive(test_viewer: TestViewer) -> Experiment:
    """
    将某次 test 对应 commit 的文件打包，相关命令为
        git archive -o <filename> <commit-id>
    :param test_viewer:
    :return:
    """
    repo = test_viewer.repo
    commit = Commit(repo, hex_to_bin(test_viewer.json_info['commit_hash']))

    old_path = os.getcwd()
    os.chdir(commit.tree.abspath)
    exp = Experiment('Archive')

    revert_path = exp.makedir('archive')
    revert_fn = os.path.join(revert_path, "file.zip")
    exp.regist_plugin('archive', {'file': revert_fn,
                                  'test_name': test_viewer.test_name})
    with open(revert_fn, 'wb') as w:
        commit.repo.archive(w, commit)

    exp.end()
    os.chdir(old_path)
    return exp


def reset(test_viewer: TestViewer) -> Experiment:
    """
    将工作目录中的文件恢复到某个commit
        恢复快照的 git 流程：
            git branch experiment
            git add . & git commit -m ... // 保证文件最新，防止冲突报错，这一步由 Experiment() 代为完成
            git checkout <commit-id> // 恢复文件到 <commit-id>
            git checkout -b reset // 将当前状态附到新的临时分支 reset 上
            git branch experiment // 切换回 experiment 分支
            git add . & git commit -m ... // 将当前状态重新提交到最新
                // 此时experiment 中最新的commit 为恢复的<commit-id>
            git branch -D reset  // 删除临时分支
            git branch master // 最终回到原来分支，保证除文件变动外git状态完好
    :param test_viewer: TestViewer
    :return:
    """
    repo = test_viewer.repo
    commit = Commit(repo, hex_to_bin(test_viewer.json_info['commit_hash']))

    old_path = os.getcwd()
    os.chdir(commit.tree.abspath)
    exp = Experiment('Reset')

    repo = commit.repo  # type:Repo
    with branch(commit.repo, _GITKEY.thexp_branch) as new_branch:
        repo.git.checkout(commit.hexsha)
        repo.git.checkout('-b', 'reset')
        repo.head.reference = new_branch
        repo.git.add('.')
        ncommit = repo.index.commit("Reset from {}".format(commit.hexsha))
        repo.git.branch('-d', 'reset')
    exp.regist_plugin('reset', {
        'test_name': test_viewer.test_name,  # 从哪个状态恢复
        'from': exp.commit.hexsha,  # reset 运行时的快照
        'where': commit.hexsha,  # 恢复到哪一次 commit，是恢复前的保存的状态
        'to': ncommit.hexsha,  # 对恢复后的状态再次进行提交，此时 from 和 to 两次提交状态应该完全相同
    })

    exp.end()
    os.chdir(old_path)
    return exp
