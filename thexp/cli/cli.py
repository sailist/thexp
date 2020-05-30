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
 thexp board [--host=<host>] [--port=<port>]
 thexp find <test_name>
 thexp find -t <test_name>
 thexp find -e <exp_name>
 thexp find -p <proj_name>
 thexp reset <test_name>
 thexp archive <test_name>
"""

# 检测是否init，没有报错 fatal: unable to read config file '.git/config': No such file or directory
from docopt import docopt
from typing import List
from thexp import __VERSION__
from thexp.analyser.expviewer import SummaryViewer,ExpViewer,ProjViewer,TestViewer
from thexp.globals import _DLEVEL
from thexp.analyser.web.server import run
from thexp.analyser.web.util import get_ip

print(__VERSION__)
arguments = docopt(doc, version=__VERSION__)

viewer = SummaryViewer()

if arguments['init']:
    # TODO 提供一个比较好的初始化交互
    pass

elif arguments['board']:
    host = arguments.get('<host>', None)
    if host is None:
        host = get_ip()
        if host.startswith('192'):
            host = '127.0.0.1'
    port = arguments.get('<port>', '5555')
    run(host=host, port=port)
elif arguments['find']:

    if arguments['-e']:
        exp_name = arguments['<exp_name']
        ev = viewer.find(exp_name, level=_DLEVEL.exp) # type:List[ExpViewer]
    elif arguments['-p']:
        proj_name = arguments['<proj_name>']
        pv = viewer.find(proj_name, level=_DLEVEL.proj) # type:List[ProjViewer]
    else:
        test_name = arguments['<test_name>']
        tv = viewer.find(test_name, level=_DLEVEL.test) # type:List[TestViewer]

elif arguments['reset']:
    from thexp.utils.gitutils import reset

    test_name = arguments['<test_name>']
    tv = viewer.find(test_name, level=_DLEVEL.test)
    if len(tv) == 0:
        print("can't find test {}".format(test_name))
        exit(1)
    tv = tv[0]
    exp = reset(tv)
    print('reset from {} to {}'.format(exp.plugins['reset']['from'], exp.plugins['reset']['to']))


elif arguments['archive']:
    from thexp.utils.gitutils import archive

    test_name = arguments['<test_name>']
    tv = viewer.find(test_name, level=_DLEVEL.test)
    if len(tv) == 0:
        print("can't find test {}".format(test_name))
        exit(1)
    tv = tv[0]
    exp = archive(tv)
    print('archive {} to {}'.format(test_name, exp.plugins['archive']['file']))
else:
    print(arguments)
    print(arguments.items())

exit(0)
