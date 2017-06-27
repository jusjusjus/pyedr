
from . import ekg
from . import resp
import numpy as np
from scipy.optimize import fmin

pi2 = 2 * np.pi

class Coupling:
    """ Compute phase-amplitude coupling btw EKG and RESP
    Args:
        EKG: iterable of EKG signals, or ekg.Ekg object.
        RESP: iterable of RESP signals, or resp.Resp object.
            There should be the same amount of EKG and RESP signals.
        sampling_rate: sampling rate of both signals, unless objects
            are given. (default None)
    """
    
    def __init__(self, EKG, RESP, sampling_rate=None):
        if type(EKG) is not ekg.Ekg:
            assert sampling_rate is not None, "s-rate required to build ekg.Ekg object."
            self.EKG = ekg.Ekg(EKG, sampling_rate)
        else:
            self.EKG = EKG
        if type(RESP) is not resp.Resp:
            assert sampling_rate is not None, "s-rate required to build resp.Resp object."
            self.RESP = resp.Resp(RESP, sampling_rate)
        else:
            self.RESP = RESP
        assert len(self.EKG.segments) == len(self.RESP.segments), "Incompatible EKG and RESP series."
        self.EKG.get_all_R_peaks()
        self.RESP.get_all_phases()
        self.compute_coupling()

    def compute_coupling(self):
        EKG = self.EKG
        RESP = self.RESP
        phi_j = [phase[idx] for phase, idx in zip(RESP.phase, EKG.R_peaks)]
        phi_j = self.phi_j = np.concatenate(phi_j)
        rri = self.rri = np.concatenate(EKG.R_intervals)
        rph = self.rph = np.concatenate(EKG.R_peak_heights)

        self.p_rri, self.res_rri = self.fit_sinfunc(phi_j, rri)
        self.p_rph, self.res_rph = self.fit_sinfunc(phi_j, rph)

        self.compute_snr()

    def compute_snr(self):
        self.snr_rri = abs(self.p_rri[1])/self.res_rri
        self.snr_rph = abs(self.p_rph[1])/self.res_rph

    def plot(self):
        import matplotlib.pyplot as plt
        ph = np.arange(0, pi2, 0.1)
        ax = plt.subplot(211)
        plt.title("RSA-Coupling:  SNR={:.3g}".format(self.snr_rri))
        plt.plot(ph, self.sinfunc(ph, self.p_rri))
        plt.plot(self.phi_j, self.rri, 'ko')
        plt.ylabel("R-R Intervals (sec.)")
        plt.subplot(212, sharex=ax)
        plt.title("ESK-Coupling:  SNR={:.3g}".format(self.snr_rph))
        plt.plot(ph, self.sinfunc(ph, self.p_rph))
        plt.plot(self.phi_j, self.rph, 'ko')
        plt.ylabel("R-Peak Height (mV)")
        plt.xlim(0, pi2)
        plt.show()

    @staticmethod
    def sinfunc(phi, p):
        return p[0] + p[1]*np.sin(phi-p[2])

    @classmethod
    def erf(cls, p, phi, f):
        return np.mean((cls.sinfunc(phi, p)-f)**2)

    @classmethod
    def fit_sinfunc(cls, phase, f):
        p_init = [np.mean(f), np.sqrt(2)*np.std(f), np.pi]    
        p_opt = fmin(cls.erf, p_init, args=(phase, f))
        residual = np.sqrt(cls.erf(p_opt, phase, f))
        return p_opt, residual
