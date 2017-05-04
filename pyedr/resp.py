
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, detrend
from fastcache import clru_cache

pi2 = 2.0*np.pi

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
        self.compute_real_phase = True

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

    def get_prophase(self, x):
        xf = detrend(self.filter(x))
        return np.mod(np.angle(hilbert(xf)), pi2)

    def phase_from_prophase(self):
        """ Transform a pro-phase to a real phase

        The real phase has the property to rotate uniformly, leading to a
        uniform distribution density.  The prophase typically doesn't fulfill
        this property.  The following lines apply a nonlinear transformation to
        the phase signal that makes its distribution exactly uniform.
        """
        # Pool prophases from sequence
        phi = np.concatenate(self.phase)
        # Get a sorting index
        sort_idx = np.argsort(phi)
        # Get index reversing sorting
        reverse_idx = np.argsort(sort_idx)
        # Set up sorted real phase
        tht = pi2 * np.arange(phi.size)/(phi.size)
        # Reverse the sorting of it
        real_phase = tht[reverse_idx]
        # Now put in sequence again
        indices = np.cumsum([0]+[ph.size for ph in self.phase])
        self._phase = [real_phase[s:e] for s, e in zip(indices[:-1], indices[1:])]

    def get_all_phases(self):
        assert not self.phase_computed
        self._phase = [
                self.get_prophase(x)
                for x in self.segments
            ]
        self.phase_computed = True
        if self.compute_real_phase:
            self.phase_from_prophase()
