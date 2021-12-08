import numpy as np
import scipy.stats as stats


class DKL(object):
    """
    This feature compares the CDF of the initial part of the histogram to a uniform CDF
    using the D_kl metric.
    """

    def __init__(self, resolution=2, cdf_range=50, jmp_min=50, jmp_max=1000):
        # see temporal_features_calc.py for use of those fields
        self.resolution = resolution
        self.cdf_range = cdf_range
        self.jmp_min = jmp_min
        self.jmp_max = jmp_max

        self.name = 'D_KL'

    def calculate_feature(self, start_cdf=None, rhs=None, midband=None, **kwargs):
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
            start_cdf = start_band / np.sum(start_band, axis=1)
        uniform = np.ones(start_cdf.shape[1]) / start_cdf.shape[1]

        result = np.zeros((len(start_cdf), 2))

        for i, cdf in enumerate(start_cdf):
            dkl = stats.entropy(np.where(cdf > 0, cdf, 0), uniform)
            if dkl == float('inf'):
                print(cdf)
                raise AssertionError
            result[i, 0] = dkl

        if midband is None:
            assert rhs is not None
            midband = rhs[:, self.resolution * self.jmp_min: self.resolution * self.jmp_max]
            mid_cdf = midband / np.sum(midband, axis=1)
        uniform = np.ones(mid_cdf.shape[1]) / mid_cdf.shape[1]

        for i, cdf in enumerate(mid_cdf):
            dkl = stats.entropy(np.where(cdf > 0, cdf, 0), uniform)
            if dkl == float('inf'):
                print(cdf)
                raise AssertionError
            result[i, 1] = dkl

        return result

    def set_fields(self, resolution, cdf_range, jmp_min, jmp_max, **kwargs):
        self.resolution = resolution
        self.cdf_range = cdf_range
        self.jmp_min = jmp_min
        self.jmp_max = jmp_max

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["d_kl_start", "d_kl_mid"]
