import os
from datetime import datetime


def curent_date(fmt='%y-%m-%d-%H%M%S', dateobj: datetime = None):
    if dateobj is not None:
        return dateobj.strftime(fmt)
    return datetime.now().strftime(fmt)


def date_from_str(value, fmt='%y-%m-%d-%H%M%S') -> datetime:
    return datetime.strptime(value, fmt)


def file_atime2date(file, fmt='%y%m%d-%H%M%S') -> str:
    """获取某文件的 atime """
    return curent_date(fmt, datetime.fromtimestamp(os.path.getatime(file)))