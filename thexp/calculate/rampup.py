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
# TODO

import numpy as np


def sigmoid_rampup(current, rampup_length=100, top=1,bottom=0,reverse=False):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length=100, start=0, end=1):
    """
    :param current:
    :param rampup_length:
    :param start:
    :param end:
    :return:
    """

    percent = current / rampup_length
    if percent > 1:
        return end
    else:
        return percent * end + (1 - percent) * start


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    # if current > rampdown_length:
    #     return float(.5 * (np.cos(np.pi) + 1))
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
