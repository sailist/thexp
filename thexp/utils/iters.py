"""
Methods about iterator
"""
from collections.abc import Iterator


def deep_chain(item):
    """flatten iterator"""
    if isinstance(item, Iterator):
        for i in item:
            if isinstance(i, Iterator):
                for ii in deep_chain(i):
                    yield ii
            else:
                yield i
    else:
        yield item


def is_same_type(items, ty=None):
    for item in items:
        if ty is None:
            ty = type(item)
        else:
            if type(item) != ty:
                return False
    return True
