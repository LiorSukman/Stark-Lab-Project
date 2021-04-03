import numpy as np
import math

# TODO fix all descriptions

class RiseTime(object):
    """
    TODO add description
    """

    def __init__(self, resolution=2, cdf_range=30):
        self.resolution = resolution
        self.cdf_range = cdf_range

        self.name = 'rise time'

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
        ach_rise_time = (start_cdf > 1 / math.e).argmax()

        return [[ach_rise_time]]

    def set_fields(self, resolution, cdf_range, **kwargs):
        self.resolution = resolution
        self.cdf_range = cdf_range

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['rise_time']
