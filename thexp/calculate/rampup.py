"""

"""

import numpy as np
# from ..utils.decorators.deprecated import deprecated
from thexp import __VERSION__


# @deprecated(deprecated_in='1.4.0.11',
#             removed_in='1.5',
#             current_version=__VERSION__,
#             details='There is now a more suitable function and classes in thexp.calculate.schedule')
def sigmoid_rampup(current, rampup_length=100, top=1, bottom=0, reverse=False):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


# @deprecated(deprecated_in='1.4.0.11',
#             removed_in='1.5',
#             current_version=__VERSION__,
#             details='There is now a more suitable function and classes in thexp.calculate.schedule')
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


# @deprecated(deprecated_in='1.4.0.11',
#             removed_in='1.5',
#             current_version=__VERSION__,
#             details='There is now a more suitable function and classes in thexp.calculate.schedule')
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    # if current > rampdown_length:
    #     return float(.5 * (np.cos(np.pi) + 1))
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
