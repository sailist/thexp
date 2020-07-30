"""

"""

import sys
sys.path.insert(0,'./')
from thexp import __VERSION__
print(__VERSION__)

from thexp.analyser.web import server
server.run()
