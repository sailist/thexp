"""

"""

import socket

_ip = None
def get_ip() -> str:
    # global _ip
    # if _ip is not None:
    #     return _ip

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        _ip = s.getsockname()[0]
    finally:
        s.close()

    if isinstance(_ip,str):
        if _ip.startswith('192'):
            _ip = '127.0.0.1'

    return _ip