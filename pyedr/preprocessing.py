import numpy as np
import pylab as plt
from scipy import signal

def splitted(segments, avg_segment_len=2500):
    splitted_segments = []
    for segment in segments:
        segment_length = segment.shape[1]
        sections = np.linspace(0, segment_length,
                               segment_length//avg_segment_len,
                               endpoint=False, dtype=int)[1:]
        if len(sections):
            splitted_segments.extend(np.split(segment, sections, 1))
        else:
            splitted_segments.append(segment)
    return splitted_segments


class Preprocess:
    """ preprocessing of the ecg/resp signals
    
    Attributes:
    """
    def __init__(self, *_,
                 rdp=False,
                 bandpass_frequencies=(0.1, 10),
                 original_frequency=250,
                 resampling=10):
        self._use_rdp = rdp
        self._bandpass_frequencies = bandpass_frequencies
        self._resampling = resampling
        self._original_frequency = original_frequency
    def _normalized_std(self, x):
        return (x - np.mean(x)) / np.std(x)

    def _filtered(self, x):
        nyquist_frequency = self._original_frequency / 2
        b, a = signal.butter(
            2, [f/nyquist_frequency for f in self._bandpass_frequencies], 
            output='ba', btype='bandpass')
        return signal.filtfilt(b, a, x, padlen=30, padtype='even')

    def _normalized(self, x):
        x = self._normalized_std(x)
        x = self._filtered(x)
        x = self._resampled(x)
        return x

    def _resampled(self, x):
        return x[::self._resampling]

    def _rdp(self, input_, time, target=None):
        #        u = np.linspace(0, spline[0][-1], max(2, 100*spline[0][-1]))
        #stroke = np.array(interpolate.splev(u, spline)).transpose()
        stroke = rdp(np.column_stack([time, input_]), epsilon=0.01)

        input_rdp, time_rdp = stroke[:, 1], stroke[:, 0]
        if target is not None:
            target_rdp = target[np.searchsorted(time, time_rdp)]
        else:
            target_rdp = None
        print("reduces from {} to {}".format(len(time), len(time_rdp)))
        return input_rdp, time_rdp, target_rdp

    def __call__(self, time, input_, target=None):
        norm_input = self._normalized(input_)
        if target is not None:
            norm_target = self._normalized(target)
        else:
            norm_target = None
        norm_time = self._resampled(time)
        if self._use_rdp:
            norm_input, norm_time, norm_target = \
                self._rdp(norm_input, norm_time, norm_target)

        return norm_time, norm_input, norm_target
