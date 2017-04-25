
import numpy as np
import matplotlib.pyplot as plt
from fastcache import clru_cache
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple, ChainMap

@clru_cache(maxsize=256)
def sr2sqdt(sampling_rate):
    return np.sqrt(1.0/np.float64(sampling_rate))

__all__ = ['SyntheticECG']

def random_walk(N, sampling_rate):
    """ generate random walk
    Args:
        time (array): times (must be equidistant a la np.linspace)
    Returns:
        array of len like time
    """
    return np.sr2sqdt(sampling_rate) * np.random.randn(N).cumsum()


Signal = namedtuple("Signal", ["time", "input", "target"])

class SignalGenerator:
    """ Interface for Signal
    """
    def __init__(self, max_time=10, sampling_rate=10, seed=42):
        self._sampling_rate = np.int64(sampling_rate)
        self._time = np.linspace(0, max_time, max_time * self._sampling_rate)
        self._max_time = max(self._time)
        np.random.seed(seed)

    def __call__(self):
        """ generate the new Signal
        """
        raise NotImplementedError

    def show_single_instance(self, signal = None):
        if signal is None:
            signal = self()
        ax = plt.subplot(211)
        plt.plot(signal.time, signal.input)
        plt.subplot(212, sharex=ax)
        plt.plot(signal.time, signal.target)
        plt.xlabel("Time (sec.)")
        plt.tight_layout()
        plt.show()


class VanillaECGGenerator(SignalGenerator):
    """ Prototype of a ECG signal generator providing essential parameters

    example:
        ECG().show_single_instance()
    """

    def __init__(self,
                 heart_rate=60/60, respiration_rate=15/60,
                 respiration_rate_stdev = 1/60,
                 respiration_fluctuations=1/60,
                 esk_strength=0.1, rsa_strength=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heart_rate = heart_rate
        self._respiration_rate = respiration_rate
        self._respiration_fluctuations = respiration_fluctuations
        self._respiration_rate_stdev = respiration_rate_stdev
        self._rsa_strength = rsa_strength
        self._esk_strength = esk_strength

class SimpleECGGenerator(VanillaECGGenerator):
    """ Simple ECG signal generator based on single peaks in ECG
    """
    def __init__(self,
                 heart_fluctuations=0.1,
                 heart_rate_stdev = 20/60,

                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heart_fluctuations = heart_fluctuations
        self._heart_rate_stdev = heart_rate_stdev

    @property
    def _time_diff(self):
        return 1 / self._sampling_rate

    def _gen_phase_heartbeat(self):
        frequency = np.random.normal(self._heart_rate,
                                     self._heart_rate_stdev)
        return (np.random.random() +
                self._time * frequency +
                self._heart_fluctuations * random_walk(len(self._time), self._sampling_rate))

    def _gen_phase_respiration(self):
        frequency = np.random.normal(self._respiration_rate,
                                     self._respiration_rate_stdev)
        return (np.random.random() +
                self._time * frequency +
                self._respiration_fluctuations * random_walk(len(self._time), self._sampling_rate))

    def _gen_respiration(self, phase):
        return np.sin(2 * np.pi * phase)

    def _coupled_via_esk(self, heartbeat, respiration):
        return heartbeat * (1 + self._esk_strength * respiration)

    def _couple_via_rsa(self, phase_heartbeat, phase_respiration):
        """ including respiratory sinus arrhythmia
        """
        phase_heartbeat[:] += (self._time_diff * self._rsa_strength *
            np.sin(2 * np.pi * phase_respiration)).cumsum()

    def _gen_heartbeat(self, phase):
        WIDTH = 0.06
        theta = phase % 1
        delta_theta = theta - 0.5
        return np.exp(-delta_theta**2 / (2 * WIDTH))

    def __call__(self):
        phase_respiration = self._gen_phase_respiration()
        phase_heartbeat = self._gen_phase_heartbeat()
        self._couple_via_rsa(phase_heartbeat, phase_respiration)
        heartbeat = self._gen_heartbeat(phase_heartbeat)
        respiration = self._gen_respiration(phase_respiration)
        heartbeat_signal = self._coupled_via_esk(heartbeat, respiration)
        return Signal(time=self._time, input=heartbeat_signal,
                      target=respiration)


class Respiration:

    def __init__(self, ensemble_mean_period_duration, stdev_period_duration):
        mean_period_duration = np.random.normal(ensemble_mean_period_duration, stdev_period_duration)
        period_durations = np.random.normal( mean_period_duration, stdev_period_duration, 1000)
        self._respiration_start_times = np.concatenate([[0], period_durations.cumsum()])

    def __call__(self, time):
        time = time  +  100
        idx = np.searchsorted(self._respiration_start_times, time)
        period_duration = (self._respiration_start_times[idx] -
                           self._respiration_start_times[idx - 1])
        phase = (time - self._respiration_start_times[idx - 1]) / period_duration
        phase = phase % 1
        return np.exp(-(phase - 0.5)**2 / (2 * 0.03))




class SyntheticECGGenerator(VanillaECGGenerator):
    """ Generate synthetic ECG Signals

    >>> get_signal = synthetic.SyntheticECG()
    >>> signal = get_signal()
    >>> time, input_, target = signal

    Paper: P. McSharry el. al.,
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 50, NO. 3, MARCH 2003

    Args:
        max_time: length of sequence in sec (default 10)
        sampling_rate: samples per second (default 250)
        heart_rate: heartrate in beats per second  (default 60/60)
        respiration_rate: ensemble average of the respiration rate
            in beats per second (default 15/60)
        respiration_rate_stdev: the standard deviation of the
            respiration rate (default 1/60) for both the ensemble fluctuations
            and the insample fluctuations
        esk_strength: ESK coupling stength, the ecg signal varies with
            1 + esk_strength * respiration   (default 0.1)
        rsa_strength: strength of the respiratory sinus arrhythmia (default 0.5)
        heart_noise_stength: noise added to ecg signal (default 0.05)
        heart_rate_fluctuations: relative heat rate fluctuations (default 0.05)
        respiration_noise_stength: noise added to respiration signal
            (default 0.05)
        esk_spot_width (None or float): if None then the esk acts uniform if float
            than only in a spot centered at the R-peak with the given width
            (see also show_single_trajectory)
        rsa_spot_width (None or float): if None then the esk acts uniform if float
            than it acts everywhere except in the spot centered at the R-peak 
            with the given width (see also show_single_trajectory)
    """
    default_params = {
            'sampling_rate': 250,
            'heart_noise_strength': 0.05,
            'respiration_noise_strength': 0.05,
            'esk_spot_width': 0.15,
            'rsa_spot_width': 0.25
    }

    PeakParameter = namedtuple("Parameter", "a b theta")
    PEAK_PARAMETERS = {
        "P": PeakParameter(a=1.2,  b=.25, theta=-np.pi/3),
        "Q": PeakParameter(a=-5.0, b=.1,  theta=-np.pi/12),
        "R": PeakParameter(a=30.0, b=.1,  theta=0),
        "S": PeakParameter(a=-7.5, b=.1,  theta=np.pi/12),
        "T": PeakParameter(a=.75,  b=.4,  theta=np.pi/2)
    }

    A = 0.0
    ESK_SPOT_WIDTH_QRS = 2 * np.pi / 12
    RSA_SPOT_WIDTH_QRS = 2 * np.pi / 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(**ChainMap(kwargs, self.default_params))

    def _random_process(self, coefficients, time):
        #TODO(dv): ideally this should be a Ornstein-Uhlenbeck process, 
        #          how is its spectral decomposition?
        return sum(c * np.sin((k + 1) * np.pi * time / self._max_time / 2) / (k + 1) 
                   for k, c in enumerate(coefficients)) / np.pi

    def _dynamical_equation_of_motion(self, state, time, respiration,
                                      random_coefficients):
        x, y, z = state

        alpha = 1 - np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        r = respiration(time)
        omega_heart_mean = 2 * np.pi * self._heart_rate
        omega_heart = omega_heart_mean * (1 + 
            self._heart_rate_fluctuations * self._random_process(
                random_coefficients, time))
        omega_heart = max(omega_heart, 0)
        if self._rsa_spot_width is None:
            omega_heart = omega_heart * (1 + self._rsa_strength * r)
        else:
            omega_heart = (omega_heart * 
             (1 + (1 - np.exp(- theta**2 / self._rsa_spot_width**2)) *  
                   self._rsa_strength * r))
        
        x_dot = alpha * x - omega_heart * y
        y_dot = alpha * y + omega_heart * x

        z_dot = -z + self.A * r
        for a_i, b_i, theta_i in self.PEAK_PARAMETERS.values():
            delta_theta = (theta - theta_i + np.pi) % (2 * np.pi) - np.pi
            z_dot += -(omega_heart / omega_heart_mean * a_i * delta_theta *
                       np.exp(-delta_theta**2 / (2 * b_i**2)))

        return [x_dot, y_dot, z_dot]

    def show_single_trajectory(self):
        null_resp = lambda x: np.zeros_like(x)
        trajectory = self._heartbeat_trajectory(respiration=null_resp)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        theta = self._time
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]
        theta = np.arctan2(y, x)
        z2 = np.exp(- theta**2 / self._esk_spot_width**2)
        ax.plot(x, y, z)
        ax.plot(x, y, z2 / 10)
        plt.show()

    @staticmethod
    def _noise(signal, stdev):
        signal += np.random.normal(0, stdev, len(signal))

    def _respiration(self):
        return Respiration(1 / self._respiration_rate,
                           self._respiration_rate_stdev / self._respiration_rate**2)

    def _coupled_via_esk(self, heartbeat_trajectory, respiration):
        heartbeat = heartbeat_trajectory[:, 2]
        if self.esk_spot_width is None:
            heartbeat[:] = heartbeat * (1 + self._esk_strength * respiration)
        else:
            x = heartbeat_trajectory[:, 0]
            y = heartbeat_trajectory[:, 1]
            theta = np.arctan2(y, x)
            heartbeat[:] *= (1 + np.exp(- theta**2 / self._esk_spot_width**2) * 
                                 self._esk_strength * respiration)

    def _heartbeat(self, respiration):
        return self._heartbeat_trajectory(respiration)[:, 2]

    def _heartbeat_trajectory(self, respiration):
        random_coefficients = np.random.normal(size=30)
        trajectory = odeint(
            self._dynamical_equation_of_motion,
            [-1, 0, 0], np.concatenate([np.linspace(-5, -0.1, 10), self._time]),
            (respiration, random_coefficients))[10:]
        trajectory[:, 2] *= 20
        return trajectory

    def __call__(self):

        respiration_t =  self._respiration()
        respiration = respiration_t(self._time)
        heartbeat_trajectory = self._heartbeat_trajectory(respiration_t)
        heartbeat = heartbeat_trajectory[:, 2]
        self._coupled_via_esk(heartbeat_trajectory, respiration)
        self._noise(heartbeat, self._heart_noise_stength)
        self._noise(respiration, self._respiration_noise_stength)
        return Signal(time=self._time, input=heartbeat, target=respiration)



if __name__ == "__main__":
    get_signal = SyntheticECGGenerator(sampling_rate=250, max_time=10)
    get_signal.show_single_instance()
