import numpy as np
import scipy.signal as signal


class PSD(object):
    """
    This feature performs power spectral analysis on the histogram calculating the centroid of the power spectral
    density and the centroid of its derivative
    """

    def __init__(self):
        self.name = 'Power Spectral Density'

    def calculate_feature(self, rhs, **kwargs):
        """
        inputs:
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the start_cdf if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated measurements of the feature value as described before.
        """
        f, pxx = signal.periodogram(rhs, 20_000)
        centroid = np.sum(f * pxx) / np.sum(pxx)  # TODO maybe it should be the || of pxx

        der_pxx = np.abs(np.gradient(pxx))  # TODO check if there really can be negative values here
        der_centroid = np.sum(f * der_pxx) / np.sum(der_pxx)  # TODO maybe it shoud be the || of pxx

        return [[centroid, der_centroid]]

    def set_fields(self, **kwargs):
        pass

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["psd_center", "der_psd_center"]
