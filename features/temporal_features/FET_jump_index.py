import numpy as np


class Jump(object):
    """
    This feature compares the middle band of the histogram to a linear change.
    """

    def __init__(self, resolution=2, mid_band_start=50, mid_band_end=1000):
        # see temporal_features_calc.py for use of those fields
        self.resolution = resolution
        self.jmp_min = mid_band_start
        self.jmp_max = mid_band_end

        self.name = 'jump index'

    def calculate_feature(self, mid_band=None, rhs=None, **kwargs):
        """
        inputs:
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the long-band if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated feature value as described before.
        """
        if mid_band is None:
            assert rhs is not None
            mid_band = rhs[:, self.resolution * self.jmp_min: self.resolution * self.jmp_max + 1]

        mid_cdf = (np.cumsum(mid_band, axis=1).T / np.sum(mid_band, axis=1)).T
        uniform_cdf = np.linspace(0, 1, mid_cdf.shape[1])

        result = abs((mid_cdf - uniform_cdf)).sum(axis=1) / mid_cdf.shape[1]

        """
        result = np.zeros((len(mid_band), 1))
        for i, mid in enumerate(mid_band):
            jmp_line = np.linspace(mid[0], mid[-1], len(mid))
            # TODO after assuring this is ok change 5000 to number of samples and make the 50 part of the class
            ach_jmp = np.sum((mid - jmp_line) ** 2)
            result[i, 0] = ach_jmp

        return result"""

        return np.expand_dims(result, axis=1)

    def set_fields(self, resolution, mid_band_start, mid_band_end, **kwargs):
        self.resolution = resolution
        self.jmp_min = mid_band_start
        self.jmp_max = mid_band_end

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['jump']
