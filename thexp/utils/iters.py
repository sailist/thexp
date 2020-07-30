from collections.abc import Iterator


def iter2pair(obj):
    """"""
    for k in obj:
        if isinstance(obj, dict):
            yield k, obj[k]
        elif isinstance(k, (list, tuple)):
            yield k[0], k[1]
        elif isinstance(k, dict):
            for kk, vv in k.items():
                yield kk, vv


def deep_chain(item):
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