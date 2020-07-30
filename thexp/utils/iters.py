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
    from collections.abc import Iterable
    if isinstance(item, Iterable):
        for i in item:
            if isinstance(i, Iterable):
                for ii in deep_chain(i):
                    yield ii
            else:
                yield i
    else:
        yield item