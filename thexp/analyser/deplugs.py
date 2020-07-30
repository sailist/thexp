"""

"""
from thexp.analyser.expviewer import TestViewer

deplug_map = {}


def deplug(cls):
    deplug_map[cls.__plugin_name__] = cls()
    return cls


class Deplug:
    __plugin_name__ = None

    def __call__(self, tv: TestViewer, value: dict) -> dict:
        """
        默认行为：返回插件注册时候添加的信息
        Args:
            tv:
            value:

        Returns:

        """
        return tv.get_plugin(self.__plugin_name__)


@deplug
class DeParams(Deplug):
    __plugin_name__ = 'params'

    def __call__(self, tv: TestViewer, value: dict) -> dict:
        return tv.params.inner_dict.jsonify()


@deplug
class DeReset(Deplug):
    __plugin_name__ = 'reset'


@deplug
class DeArchive(Deplug):
    __plugin_name__ = 'archive'


@deplug
class DeTrainer(Deplug):
    __plugin_name__ = 'trainer'


@deplug
class DeWriter(Deplug):
    __plugin_name__ = 'writer'


@deplug
class DeLogger(Deplug):
    __plugin_name__ = 'logger'
