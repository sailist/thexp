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
import os
import sys

from git import Git, Repo

hgit = Git()

py_gitignore = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

.idea/
.thexp/"""


def git_config_syntax(value: str):
    return value.replace('\\\\', '/').replace('\\', '/')


def git_root(dir="./"):
    """
    判断某目录是否在git repo 目录内（包括子目录）
    :param dir:  要判断的目录。默认为程序运行目录
    :return: 如果是，返回该repo的根目录（包含 .git/ 的目录）
        否则，返回空
    """
    cur = os.getcwd()
    os.chdir(dir)
    try:
        res = Git().execute("git rev-parse --git-dir")
    except Exception as e:
        print(e)
        res = None
    os.chdir(cur)
    return res


class branch:
    def __init__(self, repo: Repo, branch):
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.repo.head.reference = self.old_branch
        # self.repo.head.reset(index=True, working_tree=True)


section_name = 'thexp'
thexp_branch = 'experiment'


def load_repo():
    path = git_root()
    if path is None:
        print("fatal: not a git repository (or any of the parent directories)")
        print("-----------------------")
        path = input("type root path to init this project, (default: {})".format(os.getcwd()))
        res = Repo.init(path)
        res.index.add("*")
        res.index.commit("initial commit")
    else:
        res = Repo(path)
    _check_git_config(res)
    return res


def renormpath(path):
    return os.path.normcase(path).replace("\\", '/')


def _check_git_config(repo):
    writer = repo.config_writer()
    if not writer.has_section(section_name):
        writer.add_section(section_name)

    if writer.get_value(section_name, "expsdir", "") == '':
        writer.add_value(section_name, "expsdir",
                         renormpath(os.path.join(repo.working_dir, '.thexp/experiments')).replace("\\", '/'))

    if writer.get_value(section_name, "projname", "") == '':
        writer.add_value(section_name, "projname", renormpath(os.path.basename(repo.working_dir)))

    ignorefn = os.path.join(repo.working_dir, ".gitignore")
    if not os.path.exists(ignorefn):
        with open(ignorefn, 'w', encoding='utf-8') as w:
            w.write(py_gitignore)
    else:
        have_proj = False
        with open(ignorefn, 'r', encoding='utf-8') as r:
            for line in r:
                if line.strip() == '.thexp/':
                    have_proj = True
                    break
        if not have_proj:
            with open(ignorefn, 'a', encoding='utf-8') as w:
                w.write("\n.thexp/\n")

    writer.write()
    writer.release()


from .generel_util import curent_date

import json

commit = None
commit_time = None

repo = load_repo()
def check_commit_exp():
    global commit,commit_time
    if commit is None:
        with branch(repo, thexp_branch):
            repo.index.add("*")
            commit = repo.index.commit(json.dumps(commit_info,indent=2))
            commit_time = curent_date()

commit_info = dict(
        date=curent_date(),
        args=sys.argv,
        environ="jupyter" if "jupyter_core" in sys.modules else "python",
        version=sys.version,
)


