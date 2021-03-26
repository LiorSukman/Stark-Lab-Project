import numpy as np
import scipy.signal as signal

# TODO fix all descriptions

class PSD(object):
    """
    TODO add description
    """

    def __init__(self):
        pass

    def calculate_feature(self, rhs, **kwargs):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        f, pxx = signal.periodogram(rhs, 20_000)
        centeroid = np.sum(f * pxx) / np.sum(pxx)  # TODO maybe it shoud be the || of pxx

        der_pxx = np.abs(np.gradient(pxx))  # TODO check if there really can be negative values here
        der_centeroid = np.sum(f * der_pxx) / np.sum(der_pxx)  # TODO maybe it shoud be the || of pxx

        return centeroid, der_centeroid

    def set_fields(self, **kwargs):
        pass

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["psd_center", "der_psd_center"]
