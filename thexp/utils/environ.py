import os
from sys import platform

class ENVIRON_:
    IS_WIN = (platform == "win32")
    IS_MAC = (platform == "darwin")
    IS_LINUX = (platform == "linux" or platform == "linux2")

    IS_REMOTE = any([i in os.environ for i in ['SHELL',
                                               'SHLVL',
                                               'SSH_CLIENT',
                                               'SSH_CONNECTION',
                                               'SSH_TTY']])
    IS_LOCAL = not IS_REMOTE

    IS_PYCHARM = os.environ.get("PYCHARM_HOSTED", 0) == "1"
    IS_PYCHARM_DEBUG = eval(os.environ.get('IPYTHONENABLE', "False"))
