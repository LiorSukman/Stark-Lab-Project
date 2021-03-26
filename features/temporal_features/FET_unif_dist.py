import numpy as np


# TODO fix all descriptions

class UnifDist(object):
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
        uniform_cdf = np.linspace(0, 1, len(start_cdf))
        unif_dist = (start_cdf - uniform_cdf) / len(start_cdf)
        return unif_dist

    def set_fields(self, resolution, cdf_range, **kwargs):
        self.resolution = resolution
        self.cdf_range = cdf_range

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['unif_dist']
