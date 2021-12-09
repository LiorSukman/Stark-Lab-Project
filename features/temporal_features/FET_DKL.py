import numpy as np
import scipy.stats as stats


class DKL(object):
    """
    This feature compares the CDF of the initial part of the histogram to a uniform CDF
    using the D_kl metric.
    """

    def __init__(self, resolution=2, start_band_range=50, mid_band_start=50, mid_band_end=1000):
        # see temporal_features_calc.py for use of those fields
        self.resolution = resolution
        self.start_band_range = start_band_range
        self.mid_band_start = mid_band_start
        self.mid_band_end = mid_band_end

        self.name = 'D_KL'

    def calculate_feature(self, start_band=None, rhs=None, midband=None, **kwargs):
        """
        inputs:
        start_cdf: One dimensional ndarray. Starting part of the cumulative distribution function
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the start_cdf if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated feature value as described before.
        """
        if start_band is None:
            assert rhs is not None
            start_band = rhs[:, :self.resolution * self.start_band_range]
        start_band_dens = (start_band.T / np.sum(start_band, axis=1)).T
        uniform = np.ones(start_band.shape[1]) / start_band.shape[1]

        result = np.zeros((len(start_band), 2))

        for i, dens in enumerate(start_band_dens):
            dkl = stats.entropy(dens, uniform)
            if dkl == float('inf'):
                print(dens)
                raise AssertionError
            result[i, 0] = dkl

        if midband is None:
            assert rhs is not None
            midband = rhs[:, self.resolution * self.mid_band_start: self.resolution * self.mid_band_end + 1]
        mid_dens = (midband.T / np.sum(midband, axis=1)).T
        uniform = np.ones(mid_dens.shape[1]) / mid_dens.shape[1]

        for i, dens in enumerate(mid_dens):
            dkl = stats.entropy(dens, uniform)
            if dkl == float('inf'):
                print(dens)
                raise AssertionError
            result[i, 1] = dkl

        return result

    def set_fields(self, resolution, start_band_range, mid_band_start, mid_band_end, **kwargs):
        self.resolution = resolution
        self.start_band_range = start_band_range
        self.mid_band_start = mid_band_start
        self.mid_band_end = mid_band_end

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["d_kl_start", "d_kl_mid"]
