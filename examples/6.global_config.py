"""

"""
import sys
sys.path.insert(0,"../")
from thexp import __VERSION__
print(__VERSION__)

from thexp import Experiment,glob

exp = Experiment("expname")

# glob.add_value("key",'value','user')
# glob.add_value("key",'value','exp')
# glob.add_value("key",'value','repository')
glob['a'] = 4

from pprint import pprint
pprint(glob.items())
pprint(exp.config_items())