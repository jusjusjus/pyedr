
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, detrend
from fastcache import clru_cache

@clru_cache(maxsize=256)
def low_pass_filter(f_max, sampling_rate):
        return butter(4, 2.0 * f_max/float(sampling_rate), btype='low')

class Resp:

    low_pass_frequency = 15.0
    threshold = 1.5
    rri_threshold = 0.300

    def __init__(self, segments, sampling_rate):
        self.sampling_rate = sampling_rate
        self.segments = segments
        self.num_segments = len(self.segments)
        self.phase_computed = False

    @property
    def phase(self):
        assert self.phase_computed, "Call get_all_phases() first."
        return self._phase

    def plot(self):
        import matplotlib.pyplot as plt
        signal = self.segments[0]
        phase = self.phase[0]
        t = np.arange(signal.size)/float(self.sampling_rate)
        ax = plt.subplot(211)
        plt.plot(t, signal)
        plt.subplot(212, sharex=ax)
        plt.plot(t, phase, 'k-')
        plt.show()

    def filter(self, signal):
        b, a = low_pass_filter(self.low_pass_frequency, self.sampling_rate)
        return filtfilt(b, a, signal)

    def get_phase(self, x):
        xf = detrend(self.filter(x))
        return np.mod(np.angle(hilbert(xf)), 2.0*np.pi)

    def get_all_phases(self):
        assert not self.phase_computed
        self._phase = [
                self.get_phase(x)
                for x in self.segments
                ]
        self.phase_computed = True


