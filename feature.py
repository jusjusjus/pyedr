
import os
import numpy as np
from pyedr import Dataset
from pyedr.ekg import Ekg
from pyedr.resp import Resp
from pyedr.coupling import Coupling
import matplotlib.pyplot as plt


fs = 250
def get_data():
    num_samples = fs * 30
    ds = Dataset(subject_ids=['synthetic'],
                       num_of_segments=4,
                       sampling_rate=fs,
                       num_samples=num_samples,
                       esk_strength=0.0)
    X = ds.get_data()
    EKG  = [x[0] for x in X]
    RESP = [x[1] for x in X]
    return EKG, RESP

ekg, resp = get_data()
ekg = Ekg(ekg, fs)
resp = Resp(resp, fs)
c = Coupling(ekg, resp, fs)
c.plot()
