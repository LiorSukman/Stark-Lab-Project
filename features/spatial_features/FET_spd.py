import numpy as np

from constants import TIMESTEPS, UPSAMPLE, NUM_CHANNELS

class SPD(object):
    """
    This feature calculates the numbers of channels that have crossed a certain threshold (determined by the main
     channel), and scales the values of the maximum depolarization of these channels according to the maximal
    depolarization of the main channel.
    """

    def __init__(self, ratio=0.5, mode='step'):
        self.ratio = ratio
        self.mode = mode

        self.name = 'spatial dispersion feature'

    def calculate_feature(self, spike_list):
        """
        inputs:
        spike_list: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = [self.calc_feature_spike(spike.get_data()) for spike in spike_list]
        result = np.asarray(result)

        return result

    def calc_area(self, rel_dep):
        rel_dep = np.sort(rel_dep)
        area = 0
        for i in range(NUM_CHANNELS):
            if i == 0:
                if self.mode == 'step':
                    continue
                elif self.mode == 'lin':
                    triangle = 0.5 * rel_dep[i]
                    area += triangle
                else:
                    raise NotImplementedError(f"currently not supporting spd mode {self.mode}")
            else:
                if self.mode == 'step':
                    rect = rel_dep[i - 1]
                    area += rect
                elif self.mode == 'lin':
                    rect = rel_dep[i - 1]
                    area += rect
                    triangle = 0.5 * (rel_dep[i] - rel_dep[i - 1])
                    area += triangle
                else:
                    raise NotImplementedError(f"currently not supporting spd mode {self.mode}")
        return area

    def calc_feature_spike(self, spike):
        """
        inputs:
        spike: the spike to be processed; it is a matrix with the dimensions of (NUM_CHANNELS, TIMESTEPS * UPSAMPLE)

        The function calculates the spatial dispersion of the given spike

        returns:
        a list containing the number of channels that cross the threshold and the standard deviation of 
            the spatial dispersion vector
        """
        dep = np.min(spike, axis=1)
        main_chn = (spike.max(axis=1) - spike.min(axis=1)).argmax()  # Finding the main channel
        rel_dep = dep / dep[main_chn]  # Scaling according to the main channel
        count = np.count_nonzero(rel_dep > self.ratio)
        sd = np.std(rel_dep)
        area = self.calc_area(rel_dep)
        return [count, sd, area]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area']

