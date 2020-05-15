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

doc = """
Usage:
 thexp init
 thexp config -l
 thexp config --global <name> <value> 
 thexp config --global -u <name>
 thexp config --global -l
 thexp config --local <name> <value>
 thexp config --local -u <name>
 thexp config --local -l
 thexp config --global -e
 thexp config --local -e
"""
# 检测是否init，没有报错 fatal: unable to read config file '.git/config': No such file or directory
from docopt import docopt
from thexp.frame.experiment import globs
from thexp import __VERSION__
import os
def init():
    init_dir = os.path.join(os.getcwd(),".thexp")
    os.makedirs(init_dir,exist_ok=True)
    globs.update("local","project_dir",os.path.abspath(os.getcwd()))
    if globs.local_fn != None:
        print("ok.")

def update(mode,name,val):
    globs.update(mode,name,val)

def unset(mode,name):
    globs.unset(mode,name)

def list(globa=False,local=False):
    globs.list_config(globa,local)

arguments = docopt(doc, version=__VERSION__)

if arguments["init"]:
    init()
elif arguments["config"]:
    if arguments["--global"]:
        if arguments["-l"]:
            list(globa=True)
        elif arguments["<name>"]:
            if arguments["-u"]:
                globs.unset("global",arguments["<name>"])
            else:
                globs.update("global",arguments["<name>"],arguments["<value>"])
    elif arguments["--local"]:
        if arguments["-l"]:
            list(local=True)
        elif arguments["<name>"]:
            if arguments["-u"]:
                globs.unset("local", arguments["<name>"])
            else:
                globs.update("local", arguments["<name>"], arguments["<value>"])
    elif arguments["-l"]:
        list(True,True)
else:
    print(arguments)

exit(0)