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
from ..expviewer import TestViewer

deplug_map = {}


def deplug(cls):
    deplug_map[cls.__plugin_name__] = cls()


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
