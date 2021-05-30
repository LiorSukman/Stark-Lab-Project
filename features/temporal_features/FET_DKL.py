import numpy as np
import scipy.stats as stats


class DKL(object):
    """
    This feature compares the CDF of the initial part of the histogram to a uniform CDF
    using the D_kl metric.
    """

    def __init__(self, resolution=2, cdf_range=30):
        # see temporal_features_calc.py for use of those fields
        self.resolution = resolution
        self.cdf_range = cdf_range

        self.name = 'D_KL'

    def calculate_feature(self, start_cdf=None, rhs=None, **kwargs):
        """
        inputs:
        start_cdf: One dimensional ndarray. Starting part of the cumulative distribution function
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the start_cdf if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated feature value as described before.
        """
        if start_cdf is None:
            assert rhs is not None
            start_band = rhs[:self.resolution * self.cdf_range]
            start_cdf = np.cumsum(start_band) / np.sum(start_band)
        uniform = np.ones(len(start_cdf)) / len(start_cdf)
        dkl = stats.entropy(np.where(start_cdf > 0, start_cdf, 0), uniform)  # TODO maybe on the mid-band as well?
        if dkl == float('inf'):
            print(start_cdf)
            raise AssertionError

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
