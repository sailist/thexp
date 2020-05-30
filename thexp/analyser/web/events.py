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

    这里实现了一个简单的事件系统，原本是考虑是否将事件系统应用到前端和后台的交互过程中，但暂时没有想到扩展性强的方法
    因此该文件暂时放在这里，留待日后如有需求痛点时再做思考

    即，该文件中的内容没有被使用
"""


def empty(name, *args, **kwargs):
    pass


class WebEventHandler:
    """
    Examples:
        事件注册
        handler.on(event, func)
        ...
        事件发送
        handler[event](...)
    """

    def __init__(self) -> None:
        """init"""
        self.listener = dict()

    def on(self, key, func):
        """注册监听事件"""
        self.listener[key] = func

    def emit(self, name, *args, **kwargs):
        """发出事件并得到返回结果"""
        return self.listener.get(name, empty)(name, *args, **kwargs)

    def __getattr__(self, item):
        """magic func"""

        def foo(*args, **kwargs):
            self.emit(item, *args, **kwargs)

        return foo

    def __getitem__(self, item):
        """magic func"""

        def foo(*args, **kwargs):
            self.emit(item, *args, **kwargs)

        return foo


class EVENT:
    """注册的事件"""
    event = 'event'


handler = WebEventHandler()
