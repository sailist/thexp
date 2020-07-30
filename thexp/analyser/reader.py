"""
用于读取tensorboard记录的各种数据
"""
import pprint as pp
from collections import namedtuple

from tensorboard.backend.event_processing import event_accumulator

Scalars = namedtuple('ScalarEvent', ['wall_times', 'values', 'steps'])


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
        self._reloaded = False

    def _check_reload(self):
        if not self._reloaded:
            self.ea.Reload()

    def get_scalars(self, tag):
        self._check_reload()
        wall_times, steps, values = list(zip(*self.ea.Scalars(tag)))  # 'wall_time', 'step', 'value'
        values = [float("{:.4f}".format(i)) for i in values]
        return Scalars(wall_times, values, steps)

    @property
    def scalars_tags(self):
        self._check_reload()
        return self.tags['scalars']

    @property
    def tags(self):
        self._check_reload()
        return self.ea.Tags()

    def summary(self):
        self._check_reload()
        pp.pprint(self.ea.Tags())
