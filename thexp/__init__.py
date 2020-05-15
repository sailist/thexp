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
"""

__VERSION__ = "1.2.5.4"

from .frame.databundler import DataBundler
from .frame.experiment import Experiment,glob
from .frame.rndmanager import RndManager
from .frame.logger import Logger
from .frame.meter import Meter, AvgMeter
from .frame.params import Params
from .frame.saver import Saver
from .frame.trainer import Trainer

from .frame import callbacks
from .utils import torch


