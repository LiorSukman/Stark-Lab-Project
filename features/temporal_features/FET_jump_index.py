import numpy as np

# TODO fix all descriptions

class Jump(object):
    """
    TODO add description
    """

    def __init__(self, resolution=2, jmp_min=50, jmp_max=1200):
        self.resolution = resolution
        self.jmp_min = jmp_min
        self.jmp_max = jmp_max

    def calculate_feature(self, mid_band=None, rhs=None, **kwargs):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        if mid_band is None:
            assert rhs is not None
            mid_band = rhs[self.resolution * self.jmp_min: self.resolution * self.jmp_max + 1]
        jmp_line = np.linspace(mid_band[0], mid_band[-1], len(mid_band))
        # TODO after assuring this is ok change 5000 to number of samples and make the 50 part of the class
        ach_jmp = 50 * np.log(np.sum((mid_band - jmp_line) ** 2) / 5000)

        return [[ach_jmp]]

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
