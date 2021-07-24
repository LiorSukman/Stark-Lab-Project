import numpy as np


class Jump(object):
    """
    This feature compares the middle band of the histogram to a linear change.
    """

    def __init__(self, resolution=2, jmp_min=50, jmp_max=1000):
        # see temporal_features_calc.py for use of those fields
        self.resolution = resolution
        self.jmp_min = jmp_min
        self.jmp_max = jmp_max

        self.name = 'jump index'

    def calculate_feature(self, mid_band=None, rhs=None, **kwargs):
        """
        inputs:
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the start_cdf if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated feature value as described before.
        """
        if mid_band is None:
            assert rhs is not None
            mid_band = rhs[:, self.resolution * self.jmp_min: self.resolution * self.jmp_max + 1]
        result = np.zeros((len(mid_band), 1))
        for i, mid in enumerate(mid_band):
            jmp_line = np.linspace(mid[0], mid[-1], len(mid))
            # TODO after assuring this is ok change 5000 to number of samples and make the 50 part of the class
            ach_jmp = 50 * np.log(np.sum((mid - jmp_line) ** 2) / 5000)
            result[i, 0] = ach_jmp

        return result

    def set_fields(self, resolution, jmp_min, jmp_max, **kwargs):
        self.resolution = resolution
        self.jmp_min = jmp_min
        self.jmp_max = jmp_max

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['jump']
