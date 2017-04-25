
import numpy as np
from numpy.random import normal
from namedlist import namedlist
from collections import namedtuple, ChainMap

pi  = np.pi
pi2 = 2*pi

__all__ = ['SyntheticECG']

def get_respiratory_phase(num_samples, sampling_rate, frequency=15.0/60.0, stdev_factor=0.05):
    """ 
    Returns:
        array[num_samples]: the phase (as func of time)
    """
    w  = pi2 * frequency
    # Use sqrt to properly rescale Gaussian process.
    dw = np.sqrt(w * stdev_factor)
    dt = 1/np.float64(sampling_rate)
    sqdt = np.sqrt(dt)
    t = dt * np.arange(num_samples)
    phi_init = pi2 * np.random.rand() 
    phase = phi_init + t*w + dw*sqdt*np.random.randn(num_samples).cumsum()
    return phase


class SyntheticECGGenerator:
    """ Generate synthetic ECG Signals

    >>> get_signal = synthetic.SyntheticECG()
    >>> signal = get_signal()
    >>> time, input_, target = signal

    Paper: P. McSharry el. al.,
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 50, NO. 3, MARCH 2003

    Args:
        sampling_rate: samples per second (default 250)
        num_samples:  number of samples to generate on call (default 5000)
        heart_rate: heartrate in beats per second  (default 60/60)
        hr_stdev_factor: fraction of 'heart_rate' as its variability (default 0.05)
        respiration_rate: respiration rate in beats per second (default 15/60)
        rr_stdev_factor: fraction of 'respiration_rate' as its variability (default 0.2)
        EKG_noise_strength: standard deviation of additive noise (default 0.05)
        EKG_fluctuation_strength: stdev factor for variability of EKG waves (default 1)
        RESP_noise_strength: stdev of additive noise added to respiration signal (default 0.1)
        esk_strength: ESK coupling stength, the EKG signal varies with
            1 + esk_strength * respiration   (default 0.1)
        rsa_strength: strength of the respiratory sinus arrhythmia (default 1.0)
        rsa_dispersion: slope of sensitivity increase at the RSA-sensitive part in the EKG (default 0.1).
        rsa_width_shift: width of the RSA-sensitive part in the EKG (default 0.0).
        seed:  random seed to be passed to numpy.random.seed (default None)

    kwargs:
        Additional parameters can modify the WaveParameter for 'P', 'Q', 'R', 'S', and 'T' waves.
        It should have the form '<Wave>_<parameter>=<value>'.
        parameter:
        - a: Amplitude of the wave
        - b: Half width of the peak in radian.
        - theta: Phase of the peak in radian.
        - esk: electrostatic coupling of the peak to RESP.
        - da, db, dtheta: standard deviation of peak-to-peak variability of above parameters.
    """

    Signal = namedtuple("SyntheticEKG", ["input", "target"])

    WaveParameter = namedlist(
        "Parameter", ["a", "b", "theta", "esk", 
                      "da", "db", "dtheta"])


    def __init__(self,
            sampling_rate=250,
            num_samples=5000,
            heart_rate=60.0/60.0,
            hr_stdev_factor=0.03,
            respiration_rate=15.0/60.0,
            rr_stdev_factor=0.2,
            EKG_noise_strength=0.05,
            EKG_fluctuation_strength=0.2,
            RESP_noise_strength=0.1,
            esk_strength=0.1,
            rsa_strength=1.0,
            rsa_dispersion=0.1,
            rsa_width_shift=0.0,
            seed=None,
            **kwargs):
        self.sampling_rate = sampling_rate
        self._hr_stdev_factor = hr_stdev_factor
        self.heart_rate = heart_rate
        self.respiration_rate = respiration_rate
        self.rr_stdev_factor = rr_stdev_factor
        self.EKG_noise_strength = EKG_noise_strength
        self.EKG_fluctuation_strength = EKG_fluctuation_strength
        self.RESP_noise_strength = RESP_noise_strength
        self.esk_strength = esk_strength
        self.rsa_strength = rsa_strength
        self.rsa_width_shift = rsa_width_shift
        self.rsa_dispersion = rsa_dispersion
        self.num_samples = num_samples
        self.seed = seed

        self.WAVE_PARAMETERS = {
            "P": self.WaveParameter(a= .25, b=pi2*.04,  theta=-pi/3,  esk= .5, da=0.05, db=pi2*0.002, dtheta=pi2*0.03),
            "Q": self.WaveParameter(a=-.20, b=pi2*.01,  theta=-pi/12, esk=-.5, da=0.02, db=pi2*0.001, dtheta=pi2*0.03),
            "R": self.WaveParameter(a=2.20, b=pi2*.015, theta=0,      esk= .5, da=.15,  db=pi2*0.002, dtheta=pi2*0.03),
            "S": self.WaveParameter(a=-.15, b=pi2*.01,  theta=pi/12,  esk=-.5, da=0.02, db=pi2*0.001, dtheta=pi2*0.03),
            "T": self.WaveParameter(a= .60, b=pi2*.06,  theta=pi/1.7, esk= .5, da=0.1,  db=pi2*0.002, dtheta=pi2*0.03)
            }

        for k, v in kwargs.items():
            wp_tuple = k.split('_')
            if wp_tuple[0] in self.WAVE_PARAMETERS:
                wname, pname = wp_tuple
                self.set_wave_param(wname, pname, v)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        np.random.seed(seed)

    @property
    def heart_rate(self):
        return self._heart_rate

    @heart_rate.setter
    def heart_rate(self, hr):
        self._heart_rate = hr
        self.w_heart  = pi2 * hr
        self.hr_stdev_factor = self._hr_stdev_factor

    @property
    def hr_stdev_factor(self):
        return self._hr_stdev_factor

    @hr_stdev_factor.setter
    def hr_stdev_factor(self, dhr_fac):
        self._hr_stdev_factor = dhr_fac
        # Use sqrt to properly rescale Gaussian process.
        self.dw_heart = np.sqrt(self.w_heart * dhr_fac)


    def set_wave_param(self, wave_name, param_name, val):
        setattr(self.WAVE_PARAMETERS[wave_name], param_name, val)

    def phase_deriv(self, theta, resp_state):
        """Derivative of the heartbeat phase
        Args:
            theta: heartbeat phase
            resp_state: state of the respiratory cycle (-1, 1).
                Negative values decelerate, and positive values
                accelerate the heart beat.
        General form:
            tht' = w + Q(tht, R)
            where R is the respiratory oscillation.
        Coupling function Q
            Q(tht, R) = strength R(t) / (1+exp((cos(tht)+shift)/width))
        """
        Q = self.rsa_strength/(1+np.exp((np.cos(theta)+self.rsa_width_shift)/self.rsa_dispersion)) * resp_state
        return self.w_heart + Q

    def EKG_from_phase(self, phase, RESP=None):
        """Computes EKG from a heartbeat phase timeseries
        Args:
            phase: numpy.ndarray, heartbeat phase.
            RESP: numpy.ndarray, respiratory oscillation in (-1, 1).

        RESP modulates the amplitude of each EKG wave with an absolute
        strength self.esk_strength, and a wave-specific contribution esk.
        """
        if RESP is None:
            RESP = np.zeros_like(phase, dtype=np.float64)
        assert phase.size == RESP.size
        # Local namespace is sometimes faster, often better readable
        esk_strg  = self.esk_strength
        wavep     = self.WAVE_PARAMETERS
        fluc_strg = self.EKG_fluctuation_strength
        EKG = np.zeros_like(phase, dtype=np.float64)
        for peak_idx in range(int(min(phase) / pi2) - 10, int(max(phase) / pi2) + 10):
            for a_i, b_i, tht_i, esk, da, db, dtheta in iter(wavep.values()):
                a   = normal(a_i,   fluc_strg * da)
                b   = normal(b_i,   fluc_strg * db)
                tht = normal(tht_i, fluc_strg * dtheta)
                dtht = phase - tht - peak_idx * pi2
                EKG += (1+esk_strg*esk*RESP) * a * np.exp(-dtht**2 / (2*b**2))
        return EKG

    def show_single_trajectory(self, show=False):
        import matplotlib.pyplot as plt
        trajectory = self.heartbeat_trajectory()
        heart_phase  = trajectory[:, 0]
        EKG  = trajectory[:, 1]
        RESP = trajectory[:, 2]
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(211)
        plt.plot(EKG)
        plt.subplot(212, sharex=ax)
        plt.plot(RESP)
        if show: plt.show()

    def get_resp_phase(self, num_samples):
        return get_respiratory_phase(num_samples, self.sampling_rate, self.respiration_rate, self.rr_stdev_factor)

    def heartbeat_trajectory(self):
        dt = 1./np.float64(self.sampling_rate)
        f = self.phase_deriv
        N = self.num_samples
        R = np.cos(self.get_resp_phase(N))
        dW = self.dw_heart * np.sqrt(dt) * np.random.randn(N)
        x = np.zeros((N), np.float64)
        x[0] = pi2*np.random.rand()
        for n in range(1, N):
            x[n] = x[n-1] + dt * f(x[n-1], R[n-1]) + dW[n]
        EKG  = self.EKG_from_phase(x, R)
        trajectory = np.transpose(np.vstack((x, EKG, R)))
        return trajectory

    def __call__(self):
        heartbeat_trajectory = self.heartbeat_trajectory()
        EKG  = heartbeat_trajectory[:, 1]
        RESP = heartbeat_trajectory[:, 2]
        EKG  += normal(0.0, self.EKG_noise_strength,  size=EKG.size)
        RESP += normal(0.0, self.RESP_noise_strength, size=RESP.size)
        return self.Signal(input=EKG, target=RESP)


if __name__ == "__main__":
    N = 20 * 250
    gen = SyntheticECGGenerator(sampling_rate=250, num_samples=N, rsa_strength=1)
    gen.show_single_trajectory(show=True)
