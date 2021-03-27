import numpy as np


class SPD(object):
    """
    This feature calculates the numbers of channels that have crossed a certain threshold (determined by the main
     channel), and scales the values of the maximum depolarization of these channels according to the maximal
    depolarization of the main channel.
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

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

    def calc_feature_spike(self, spike):
        """
        inputs:
        spike: the spike to be processed; it is a matrix with the dimensions of (8, 32)

        The function calculates the spatial dispersion of the given spike

        returns:
        a list containing the number of channels that cross the threshold and the standard deviation of 
            the spatial dispersion vector
        """
        dep = np.min(spike, axis=1)
        main_chn = np.argmin(spike) // 32  # Finding the main channel
        rel_dep = dep / dep[main_chn]  # Scaling according to the main channel
        count = np.count_nonzero(rel_dep > self.ratio)
        sd = np.std(rel_dep)
        return [count, sd]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['spatial_dispersion_count', 'spatial_dispersion_sd']
