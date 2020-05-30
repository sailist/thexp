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
import pprint as pp
from collections import namedtuple,defaultdict

from tensorboard.backend.event_processing import event_accumulator
Scalars = namedtuple('ScalarEvent', ['wall_times', 'values', 'steps'])
import numpy as np

class BoardReader():
    """
    ScalarEvent = namedtuple('ScalarEvent', ['wall_time', 'step', 'value'])
    CompressedHistogramEvent = namedtuple('CompressedHistogramEvent',
                                          ['wall_time', 'step',
                                           'compressed_histogram_values'])
    HistogramEvent = namedtuple('HistogramEvent',
                                ['wall_time', 'step', 'histogram_value'])
    HistogramValue = namedtuple('HistogramValue', ['min', 'max', 'num', 'sum',
                                                   'sum_squares', 'bucket_limit',
                                                   'bucket'])
    ImageEvent = namedtuple('ImageEvent', ['wall_time', 'step',
                                           'encoded_image_string', 'width',
                                           'height'])
    AudioEvent = namedtuple('AudioEvent', ['wall_time', 'step',
                                           'encoded_audio_string', 'content_type',
                                           'sample_rate', 'length_frames'])
    TensorEvent = namedtuple('TensorEvent', ['wall_time', 'step', 'tensor_proto'])
    """

    def __init__(self, file):
        self.ea = event_accumulator.EventAccumulator(file,
                                                     size_guidance={  # see below regarding this argument
                                                         event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                         event_accumulator.IMAGES: 4,
                                                         event_accumulator.AUDIO: 4,
                                                         event_accumulator.SCALARS: 0,
                                                         event_accumulator.HISTOGRAMS: 1,
                                                     })
        self.ea.Reload()

    def get_scalars(self, tag):
        wall_times, steps, values = list(zip(*self.ea.Scalars(tag)))  # 'wall_time', 'step', 'value'
        values = [float("{:.4f}".format(i)) for i in values]
        return Scalars(wall_times, values, steps)

    @property
    def scalars_tags(self):
        return self.tags['scalars']

    @property
    def tags(self):
        return self.ea.Tags()

    def summary(self):
        pp.pprint(self.ea.Tags())

