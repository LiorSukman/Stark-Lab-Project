import scipy.signal as signal
import numpy as np

from constants import UPSAMPLE

def upsample_spike(spike, upsample_factor=UPSAMPLE, fs=20_000):
    pad = (fs - spike.shape[1]) // 2
    full_spike = np.pad(spike, ((0, ), (pad, )), mode='linear_ramp')
    up_spike = signal.resample(full_spike, fs * upsample_factor, axis=1)

    start = pad * upsample_factor
    end = start + spike.shape[1] * upsample_factor

    # We copy to free the full fs * Upsample size ndarray
    return up_spike[:, start: end].copy()
