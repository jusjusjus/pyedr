#! /usr/bin/python

import os
import pyedf
import logging
import numpy as np
from scipy.signal import detrend
from .synthetic import SyntheticECGGenerator



"""Twenty young (21 - 34 years old) and twenty elderly (68 - 85 years old)
rigorously-screened healthy subjects underwent 120 minutes of continuous supine
resting while continuous electrocardiographic (ECG), and respiration signals
were collected; in half of each group, the recordings also include an
uncalibrated continuous non-invasive blood pressure signal.  Each subgroup of
subjects includes equal numbers of men and women.

All subjects remained in a resting state in sinus rhythm while watching the
movie Fantasia (Disney, 1940) to help maintain wakefulness.  The continuous
ECG, respiration, and (where available) blood pressure signals were digitized
at 250 Hz. Each heartbeat was annotated using an automated arrhythmia detection
algorithm, and each beat annotation was verified by visual inspection.

Records f1y01, f1y02, ... f1y10 and f2y01, f2y02, ... f2y10) were obtained from
the young cohort, and records f1o01, f1o02, ... f1o10 and f2o01, f2o02, ...
f2o10) were obtained from the elderly cohort.  Each group of subjects includes
equal numbers of men and women.  Each record includes ECG (with beat
annotations) and respiration, and half of those in each group (the f2* records)
include a blood pressure waveform, as noted above."""



class SubjectPrototype:

    @staticmethod
    def normalize_with_detrend(x):
        return detrend(x)

    quantiles = [0.2, 0.5, 0.8]
    @staticmethod
    def normalize_with_quantiles(x):
        x = detrend(x)
        idx = np.argsort(x)
        quant_idx = idx[[int(factor * idx.size) for factor in Subject.quantiles]]
        quantiles = x[quant_idx]
        return (x-quantiles[1])/(quantiles[2]-quantiles[0])

    @staticmethod
    def normalize_with_stdev(x):
        x = detrend(x)
        return (x-np.mean(x))/np.std(x)

    def __init__(self):
        self.normalize = {
            'stdev':     self.normalize_with_stdev,
            'quantiles': self.normalize_with_quantiles,
            'detrend':   self.normalize_with_detrend,
            None :       lambda x: x
        }

    def get_data_batches(self, sequence_len, sequences_per_batch, normalize=None):
        batch_len = sequence_len * sequences_per_batch
        data_states = self.get_data(normalize=normalize) # data[state][channel, sample]

        feature_batches, target_batches = [], []
        for data in data_states:
            features, target = data[0], data[1]
            num_samples = data.shape[1]
            num_batches = num_samples//batch_len
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_len
                batch_end   = (batch_idx+1) * batch_len

                new_shape = (sequences_per_batch, sequence_len, 1)
                feature_batch = np.reshape(features[batch_start:batch_end], new_shape)

                target_batch = np.zeros((sequences_per_batch), dtype=np.float)
                for i in range(sequences_per_batch):
                    start = batch_start + i * sequence_len
                    end   = start + sequence_len
                    target_batch[i] = np.median(target[start:end])

                feature_batches.append(feature_batch)
                target_batches.append(target_batch)

        feature_batches, target_batches = np.asarray(feature_batches), np.asarray(target_batches)
        return feature_batches, target_batches # f[batch, seq, sample, 0], t[batch, seq]

    def get_all_data_batches(self, sequence_len):
        data = self.get_all_data() # data[channel, sample]
        num_sequences = data.shape[1]//sequence_len
        num_used_samples = sequence_len * num_sequences
        data = data[:, :num_used_samples]
        feature_sequences = data[0].reshape((num_sequences, -1))
        feature_sequences = feature_sequences[:, :, None]
        targets = np.zeros((num_sequences))
        for i in range(num_sequences):
            start = i * sequence_len
            end   = start + sequence_len
            targets[i] = np.median(data[1, start:end])
        return feature_sequences, targets # out[state][feat/targ]<[seq_idx, feat_idx]/[seq_idx]>


class Subject(SubjectPrototype):
    def __init__(self, ID=None, age=None, filename=None, scorename=None):
        super().__init__()
        self.ID = ID
        self.age = age
        self.score = None
        self.recording = None
        if filename is not None:
            self.set_filename(filename)
        if scorename is not None:
            self.set_scorename(scorename)



    def set_filename(self, filename):
        assert os.path.exists(filename), filename
        self.filename = filename

    def set_scorename(self, scorename):
        assert os.path.exists(scorename), scorename
        self.scorename = scorename

    def open(self):
        if self.filename is not None:
            self.recording = pyedf.recording(self.filename)
        if self.scorename is not None:
            self.score = pyedf.Score(self.scorename)



    def get_data(self, normalize=None):
        assert self.recording is not None
        assert self.score is not None
        sampling_rate = self.recording.get_samplingrate('ECG')
        channels = ['ECG', 'RESP']
        data = []
        for state in self.score.states:
            d_s = self.recording.get_data(state_of_interest=state, channels=channels)[1]
            ekg, resp = d_s[0], d_s[1]
            ekg  = self.normalize[normalize](ekg)
            resp = self.normalize[normalize](resp)
            data.append(np.array([ekg, resp]))

        return data # data[state][channel, sample]



    def get_all_data(self):
        assert self.recording is not None
        sampling_rate = self.recording.get_samplingrate('ECG')
        state = self.recording.start
        state.duration = 60.* 60. * 2. # 2 hours
        channels = ['ECG', 'RESP']
        data = self.recording.get_data(state_of_interest=state, channels=channels)[1]
        return data # data[channel, sample]




class SyntheticSubject(SubjectPrototype):

    def __init__(self, num_of_segments=100, **kwargs):
        super().__init__()
        self.gen_signal = SyntheticECGGenerator(**kwargs)
        self.num_of_segments = num_of_segments

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def get_data(self, normalize=None):
        """
        Returns:
            data[state][channel, sample]
        """
        data = []
        for i in range(self.num_of_segments):
            ekg, resp = self.gen_signal()
            ekg  = self.normalize[normalize](ekg)
            resp = self.normalize[normalize](resp)
            data.append(np.array([ekg, resp]))
        return data

    def get_all_data(self):
        return self.get_data


class Dataset:

    DATAPATH = os.path.join('/', 'data', 'ekg', 'fantasia')
    ANOTPATH = os.path.join(os.path.dirname(__file__), 'resources', 'fantasia')

    logger = logging.getLogger(name='Dataset')

    def __init__(self, subject_ids=None, **kwargs):
        if subject_ids is None:
            subject_ids = []
        self.add_subjects(subject_ids, **kwargs)

    def id_to_subject(self, ID, **kwargs):
        if 'synthetic' in ID:
            subject = SyntheticSubject(**kwargs)
        else:
            age = 'y' if 'y' in ID else 'o'
            self.logger.debug('Adding subject "{}"'.format(ID))
            filename  = os.path.join(self.DATAPATH, ID+'.edf')
            scorename = os.path.join(self.ANOTPATH, ID+'_segments.csv')
            subject   = Subject(ID=ID, age=age, filename=filename, scorename=scorename)
        return subject

    def add_subjects(self, subject_ids, **kwargs):
        self.subjects = []
        for ID in subject_ids:
            subject = self.id_to_subject(ID, **kwargs)
            self.subjects.append(subject)

    def get_data(self, normalize=None):
        datas = []
        for subject in self.subjects:
            subject.open()
            datas.extend(subject.get_data(normalize=normalize))
        return datas # data[state][channel, sample]

    def get_data_batches(self, sequence_len, sequences_per_batch, shuffle=False, normalize=None):
        feature_batches, target_batches = [], []
        for subject in self.subjects:
            subject.open()
            data = subject.get_data_batches(sequence_len, sequences_per_batch, normalize=normalize)
            these_feature_batches, these_target_batches = data
            feature_batches.extend(these_feature_batches)
            target_batches.extend(these_target_batches)
        feature_batches = np.asarray(feature_batches)
        target_batches  = np.asarray(target_batches)

        assert len(feature_batches.shape) == 4
        (num_batches, seq_per_batch, seq_len, _) = feature_batches.shape
        assert num_batches == target_batches.shape[0]
        errmsg = "seq_per_batch={} (should be {})".format(seq_per_batch, sequences_per_batch)
        assert seq_per_batch == sequences_per_batch, errmsg
        assert seq_len == sequence_len, "seq_len={} (should be {})".format(seq_len, sequence_len)

        if shuffle:
            idx = np.random.permutation(num_batches)
            feature_batches = feature_batches[idx]
            target_batches  = target_batches[idx]

        feature_batches = np.reshape(feature_batches, (-1, sequence_len, 1))
        target_batches  = np.reshape(target_batches , (-1))

        return feature_batches, target_batches # f[seq, sample, 0], t[seq]

    def get_all_data(self):
        datas = []
        for subject in self.subjects:
            subject.open()
            datas.append(subject.get_all_data())
        return datas # data[subject][channel, sample]

    def get_all_data_batches(self, sequence_len):
        datas = []
        for subject in self.subjects:
            subject.open()
            datas.append(subject.get_all_data_batches(sequence_len))
        return datas # out[subject][feat/targ]<[seq_idx, feat_idx]/[seq_idx]>


filename = os.path.join(os.path.dirname(__file__), 'resources', 'fantasia', 'list_of_subjects.txt')
all_subject_ids = []
with open(filename, 'r') as f:
    for line in f:
        li = line.strip('\n')
        all_subject_ids.append(li)
