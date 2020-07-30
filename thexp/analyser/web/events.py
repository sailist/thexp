"""


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
