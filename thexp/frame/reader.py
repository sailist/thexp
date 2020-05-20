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

class ABSReader:
    """
    一个抽象的基类，用于表示对一个插件/模块的分析器

    :param base_dir: 相应试验的试验目录
    :param plug_info: info.json 中对应plugin（可为多个）代表的键值对字典
    """

    __plugin__ = [] # 能处理的插件，可以为复数

    def __init__(self,base_dir,plugins_dict:dict):
        """

        :param plugins_dict:
        """
        self._base_dir = base_dir
        self._plugins = plugins_dict

    def summary(self):
        """
        大致对该模块的输出信息做一个总结
        :return:
        """

        return ""


# @property
# @lru_cache()
# def board_file(self):
#     if self.has_dir('board'):
#         fs = os.listdir(os.path.join(self.test_dir, 'board'))
#         f = [i for i in fs if i.endswith('bd')][0]
#         return f
#     else:
#         return None
#
#
# @property
# @lru_cache()
# def board_reader(self):
#     if self.board_file is not None:
#         from .plotter import BoardReader
#         reader = BoardReader(self.board_file)
#         return reader
#     else:
#         return None