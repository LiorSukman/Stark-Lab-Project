import numpy as np

from constants import TIMESTEPS, UPSAMPLE, NUM_CHANNELS

class SPAT_FWHM(object):
    """
    This feature calculates the numbers of channels that have crossed a certain threshold (determined by the main
     channel), and scales the values of the maximum depolarization of these channels according to the maximal
    depolarization of the main channel.
    """

    def __init__(self, ratio=0.25, mode='step'):
        self.ratio = ratio

        self.name = 'spatial fwhm feature'

    def calculate_feature(self, spike_list, amps):
        """
        inputs:
        spike_list: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = [self.calc_feature_spike(spike.get_data(), amp) for spike, amp in zip(spike_list, amps)]
        result = np.asarray(result)

        return result

    def calc_feature_spike(self, spike, amps):
        """
        inputs:
        spike: the spike to be processed; it is a matrix with the dimensions of (NUM_CHANNELS, TIMESTEPS * UPSAMPLE)

        The function calculates the spatial dispersion of the given spike

        returns:
        a list containing the number of channels that cross the threshold and the standard deviation of
            the spatial dispersion vector
        """
        inds = amps > (self.ratio * amps)
        amps = amps[inds]
        spike = spike[inds]
        main_c_ind = amps.argmax()
        fwhm = np.sum(spike <= (np.expand_dims(spike.min(axis=1) * 0.5, axis=1)), axis=1)
        fwhm_scaled = fwhm / fwhm[main_c_ind]
        sd = np.std(fwhm_scaled)
        avg = fwhm_scaled.mean()
        return [avg, sd]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['spatial_fwhm_avg', 'spatial_fwhm_sd']

