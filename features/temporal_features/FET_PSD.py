import numpy as np
import scipy.signal as signal
from constants import UPSAMPLE


class PSD(object):
    """
    This feature performs power spectral analysis on the histogram calculating the centroid of the power spectral
    density and the centroid of its derivative
    """

    def __init__(self, resolution=2, mid_band_end=1000):
        self.resolution = resolution
        self.mid_band_end = mid_band_end

        self.name = 'Power Spectral Density'

    def calculate_feature(self, rhs, **kwargs):
        """
        inputs:
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the start_cdf if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated measurements of the feature value as described before.
        """
        result = np.zeros((len(rhs), 2))
        fs = self.mid_band_end * self.resolution * UPSAMPLE
        for i, rh in enumerate(rhs):
            rh = rh - rh.mean()
            f, pxx = signal.periodogram(rh, fs)
            inds = (f <= 100) * (f > 0)
            f, pxx = f[inds], pxx[inds]
            centroid = np.sum(f * pxx) / np.sum(pxx)  # TODO maybe it should be the || of pxx

            der_pxx = np.abs(np.gradient(pxx))  # TODO check if there really can be negative values here
            der_centroid = np.sum(f * der_pxx) / np.sum(der_pxx)  # TODO maybe it shoud be the || of pxx

            result[i, 0] = centroid
            result[i, 1] = der_centroid

        return result

    def set_fields(self, resolution, mid_band_end, **kwargs):
        self.resolution = resolution
        self.mid_band_end = mid_band_end

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["psd_center", "der_psd_center"]
