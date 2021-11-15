import numpy as np


class FWHM(object):
    """
    This feature estimates the depolarization's width using FFT
    """

    def __init__(self, fft_space=1024):
        self.fft_space = 1024

        self.name = 'width'

    def calculate_feature(self, spike_lst):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = [self.calc_feature_spike(spike.get_data()) for spike in spike_lst]
        result = np.asarray(result)
        return result

    def calc_feature_spike(self, spike):
        """
        inputs:
        spike: the spike to be processed; it is an ndarray with TIMESTEPS entries

        The function calculates the width value as described above.

        returns: a list containing the width value
        """
        # find timestamps for depolarization in ok channels, filter again to assure depolarization is reached before the
        # end
        dep = spike.min()
        # TODO Consider keeping the following lines
        # if dep_ind == len(spike) - 1:  # if max depolarization is reached at the end, it indicates noise
        #    raise Exception('Max depolarization reached at final timestamp')

        inds = spike <= self.ratio * dep
        fwhm = inds.sum()

        return [fwhm]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['fwhm']
