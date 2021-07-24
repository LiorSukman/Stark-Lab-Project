import numpy as np


class UnifDist(object):
    """
    This feature compares the starting band cumulative distribution function of the histogram to a linear (uniform)
    change.
    """

    def __init__(self, resolution=2, cdf_range=50):
        # see temporal_features_calc.py for use of those fields
        self.resolution = resolution
        self.cdf_range = cdf_range

        self.name = 'uniform distance'

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
            start_band = rhs[:, self.resolution * self.cdf_range]
            start_cdf = (np.cumsum(start_band, axis=1).T / np.sum(start_band, axis=1)).T
        uniform_cdf = np.linspace(0, 1, start_cdf.shape[1])

        unif_dist = (start_cdf - uniform_cdf).sum(axis=1) / start_cdf.shape[1]

        return np.expand_dims(unif_dist, axis=1)

    def set_fields(self, resolution, cdf_range, **kwargs):
        self.resolution = resolution
        self.cdf_range = cdf_range

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['unif_dist']
