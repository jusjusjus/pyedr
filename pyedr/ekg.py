
import numpy as np
from scipy.signal import butter, filtfilt

class Ekg:

    low_pass_frequency = 40.0
    threshold = 1.5
    rri_threshold = 0.300

    def __init__(self, segments, sampling_rate):
        self.sampling_rate = sampling_rate
        self.segments = segments
        self.num_segments = len(self.segments)
        self.get_filter()

    def plot(self):
        import matplotlib.pyplot as plt
        signal = self.segments[0]
        r_peak_idx = self.get_R_peaks(signal)
        t = np.arange(signal.size)/float(self.sampling_rate)
        plt.plot(t, signal)
        for idx in r_peak_idx:
            plt.axvline(x=t[idx])
        plt.show()

    def get_filter(self):
        self.filter_coefs = butter(4, 2.0 * self.low_pass_frequency/float(self.sampling_rate),
                btype='low')

    def filter(self, signal):
        b, a = self.filter_coefs
        return filtfilt(b, a, signal)

    def gradient(self, signal):
        return np.gradient(signal)/float(self.sampling_rate)
    
    @staticmethod
    def normalize(x):
        return (x-np.mean(x))/np.std(x)

    def get_R_peaks(self, signal):
        filtered = self.filter(signal)
        grad     = self.gradient(filtered)
        normgrad = self.normalize(grad)
        idxarr = (normgrad[1:]>self.threshold) & (normgrad[:-1]<self.threshold)
        R_peak_idx = np.arange(normgrad.size-1)[idxarr]
        RRI = (R_peak_idx[1:]-R_peak_idx[:-1])/float(self.sampling_rate)
        self.R_peak_idx = []
        for ridx, rri in zip(R_peak_idx, RRI):
            if rri > self.rri_threshold:
                self.R_peak_idx.append(ridx)
        return self.R_peak_idx

    def get_all_R_peaks(self):
        self.R_peaks = []
        for signal in self.segments:
            self.R_peaks.append(self.get_R_peaks(signal))

    def batch_data_segments(self, data, left=100, right=200):
        assert len(data) == self.num_segments, "Data ({}) not compatible with EKG segments ({}).".format(len(data), num_segments)
        data_sequences = []
        for segment, r_peaks in zip(data, self.R_peaks):
            (_, samples_in_segment) = segment.shape
            for r_peak in r_peaks:
                interval = (r_peak-left, r_peak+right)
                if interval[0]<0 or interval[1]>samples_in_segment-1:
                    continue
                seg_i = segment[:, slice(*interval)]
                data_sequences.append(seg_i)
        data_sequences = np.asarray(data_sequences)
        return data_sequences # [sequence, time, channel]
