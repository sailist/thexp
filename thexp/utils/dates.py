import os
from datetime import datetime


def curent_date(fmt='%y-%m-%d-%H%M%S', dateobj: datetime = None):
    if dateobj is not None:
        return dateobj.strftime(fmt)
    return datetime.now().strftime(fmt)


def date_from_str(value, fmt='%y-%m-%d-%H%M%S') -> datetime:
    return datetime.strptime(value, fmt)


def file_atime2date(file, fmt='%y%m%d-%H%M%S') -> str:
    """
    获取某文件的 atime
    最后一次访问时间，如果使用read函数读某个文件，会改变文件的这个时间
    """
    return curent_date(fmt, datetime.fromtimestamp(os.path.getatime(file)))


def file_mtime2date(file, fmt='%y%m%d-%H%M%S') -> str:
    """
    获取某文件的 mtime

    mtime : 最后一次修改时间，如果使用write函数写某个文件，会改变文件的这个时间

    Args:
        file:
        fmt:

    Returns:

    """
    return curent_date(fmt, datetime.fromtimestamp(os.path.getmtime(file)))