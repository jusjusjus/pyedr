
import numpy as np
from scipy.signal import butter, filtfilt
from biosppy.signals import ecg as bioecg

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
        r_peak_idx = self.R_peaks[0]
        heights = self.R_peak_heights[0]
        intervals = self.R_intervals[0]
        t = np.arange(signal.size)/float(self.sampling_rate)
        ax = plt.subplot(311)
        plt.plot(t, signal)
        for idx in r_peak_idx:
            plt.axvline(x=t[idx])
        plt.subplot(312, sharex=ax)
        plt.plot(t[r_peak_idx], heights, 'ko-')
        plt.subplot(313, sharex=ax)
        plt.plot(t[r_peak_idx], intervals, 'ko-')
        plt.show()

    def get_filter(self):
        self.filter_coefs = butter(4, 2.0 * self.low_pass_frequency/float(self.sampling_rate),
                btype='low')

    def filter(self, signal):
        b, a = self.filter_coefs
        return filtfilt(b, a, signal)

    def gradient(self, signal):
        return np.gradient(signal)/float(self.sampling_rate)
    
    def get_R_peak_heights(self, signal, r_peak_idx):
        left, right = int(0.1*self.sampling_rate), int(0.1*self.sampling_rate)
        length = signal.size
        filtered = self.filter(signal)
        heights = np.zeros((r_peak_idx.size), np.float)
        for j, r_peak in enumerate(r_peak_idx):
            l, r = max(0, r_peak-left), min(length, r_peak+right)
            segment = filtered[l:r]
            heights[j] = max(segment)-min(segment)
        return heights

    def get_R_intervals(self, r_peak_idx):
        r_intervals = np.zeros(r_peak_idx.shape, np.float)
        r_intervals[1:] = 1.0/np.float(self.sampling_rate) * (r_peak_idx[1:]-r_peak_idx[:-1])
        r_intervals[0] = r_intervals[1]
        return r_intervals
        
    def get_all_R_peaks(self, left=0, right=None):
        if right is not None and right > 0:
            right = -right
        self.R_peaks = [
            left + bioecg.ecg(signal[left:right], sampling_rate=self.sampling_rate, show=False)['rpeaks']
            for signal in self.segments
        ]
        self.R_peak_heights = [
                self.get_R_peak_heights(signal, R_peaks)
                for signal, R_peaks in zip(self.segments, self.R_peaks)
        ]
        self.R_intervals = [
                self.get_R_intervals(r_peak_idx)
                for r_peak_idx in self.R_peaks
        ]


    def batch_data_segments(self, data, left=100, right=200):
        assert len(data) == self.num_segments, "Data ({}) not compatible with EKG segments ({}).".format(len(data), num_segments)
        data_sequences = []
        for segment, r_peaks in zip(data, self.R_peaks):
            (_, samples_in_segment) = segment.shape
            for r_peak in r_peaks:
                interval = (r_peak-left, r_peak+right)
                if interval[0]<0 or interval[1]>samples_in_segment-1:
                    print('A sequence was not determined because of interval length.')
                    continue
                seg_i = segment[:, slice(*interval)]
                data_sequences.append(seg_i)
        data_sequences = np.asarray(data_sequences)
        return data_sequences # [sequence, time, channel]
