import numpy as np
import scipy.stats as stats


# TODO fix all descriptions

class DKL(object):
    """
    TODO add description
    """

    def __init__(self, resolution=2, cdf_range=30):
        self.resolution = resolution
        self.cdf_range = cdf_range

    def calculate_feature(self, start_cdf=None, rhs=None, **kwargs):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        if start_cdf is None:
            assert rhs is not None
            start_band = rhs[:self.resolution * self.cdf_range]
            start_cdf = np.cumsum(start_band) / np.sum(start_band)
        uniform = np.ones(len(start_cdf)) / len(start_cdf)
        dkl = stats.entropy(start_cdf, uniform)  # TODO maybe on the mid-band as well?

        return [[dkl]]

    def set_fields(self, resolution, cdf_range, **kwargs):
        self.resolution = resolution
        self.cdf_range = cdf_range

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["d_kl"]
