"""
"""

doc = """
Usage:
 thexp init
 thexp board [--logdir=<logdir>]
 thexp board [--test=<host>] # find test_name and tensorboard it 
 thexp board  # default open ./board
 thexp reset <test_name>
 thexp archive <test_name>
"""

# 检测是否init，没有报错 fatal: unable to read config file '.git/config': No such file or directory
from docopt import docopt

from thexp import __VERSION__
from thexp import Q

print(__VERSION__)
arguments = docopt(doc, version=__VERSION__)

if arguments['board']:
    # TODO 找到目录下的 board，用 tensorboard 打开
    #   记得确认 temp 目录

    pass

if arguments['archive']:
    test_name = arguments['<test_name>']
    query = Q.tests(test_name)
    if query.empty:
        print("can't find test {}".format(test_name))
        exit(1)
    exp = query.to_viewer().reset()

    print('archive {} to {}'.format(test_name, exp.plugins['archive']['file']))
else:
    print(arguments)
    print(arguments.items())

exit(0)
