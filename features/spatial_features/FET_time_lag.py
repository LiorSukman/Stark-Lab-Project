import numpy as np

from constants import TIMESTEPS, UPSAMPLE

# There are two options of reduction for the da vector:
# ss - sum of squares
# sa - sum of absolutes
reduction_types = ['ss', 'sa']


class TimeLagFeature(object):
    """
    This feature calculates the time difference between the main channel and all other channels in terms of
    maximal depolarization, and the following after hyperpolarization.
    The feature only takes into consideration channels that have crossed a certain threshold, determined by the
    maximal depolarization of the main channel.
    """

    def __init__(self, type_dep='ss', type_hyp='ss', ratio=0.25):
        assert type_dep in reduction_types and type_hyp in reduction_types

        # Reduction types
        self.type_dep = type_dep
        self.type_hyp = type_hyp

        # Indicates the percentage of the maximum depolarization that will be considered as a threshold
        self.ratio = ratio

        self.name = 'time lag feature'

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
        spike: the spike to be processed; it is a matrix with the dimensions of (NUM_CHANNELS, TIMESTEPS * UPSAMPLE)

        The function calculates different time lag features of the spike

        returns: a list containing the following values: -dep_red: the reduction of the depolarization vector (i.e
        the vector that indicates the time difference of maximal depolarization between each channel and the main
        channel) -dep_sd: the standard deviation of the depolarization vector -hyp_red: the reduction of the
        hyperpolarization vector - hyp_sd: the standard deviation of the hyperpolarization vector
        """
        # remove channels with lower depolarization than required
        deps = np.min(spike, axis=1)  # max depolarization of each channel
        max_dep = np.min(deps)
        fix_inds = deps <= self.ratio * max_dep
        spike = spike[fix_inds]

        # find timestamps for depolarization in ok channels, filter again to assure depolarization is reached before the
        # end
        dep_ind = np.argmin(spike, axis=1)
        # if max depolarization is reached at the end, it indicates noise
        fix_inds = dep_ind < (TIMESTEPS * UPSAMPLE - 1)
        dep_ind = dep_ind[fix_inds]
        spike = spike[fix_inds]
        if spike.shape[0] <= 1:  # if no channel passes filtering return zeros (or if only one channel)
            return [0, 0, 0, 0]

        # offset according to the main channel
        # set main channel to be the one with highest depolariztion
        main_chn = np.argmin(spike) // (TIMESTEPS * UPSAMPLE)
        dep_rel = dep_ind - dep_ind[main_chn]  # offsetting

        # calculate sd of depolarization time differences
        dep_sd = np.std(dep_rel)

        # calculate reduction
        if self.type_dep == 'ss':
            dep_red = np.mean(dep_rel ** 2)
        else:  # i.e sa
            dep_red = np.mean(np.absolute(dep_rel))

        # After wavelet transformation this is redundant
        """# find hyperpolarization indices
        hyp_ind = []
        for i, channel in enumerate(spike):
            trun_channel = channel[dep_ind[i] + 1:]
            hyp_ind.append(trun_channel.argmax() + dep_ind[i] + 1)
        hyp_ind = np.asarray(hyp_ind)

        # repeat calculations
        hyp_rel = hyp_ind - hyp_ind[main_chn]
        hyp_sd = np.std(hyp_rel)
        if self.type_hyp == 'ss':
            hyp_red = np.mean(hyp_rel ** 2)
        else:  # i.e sa
            hyp_red = np.mean(np.absolute(hyp_rel))

        return [dep_red, dep_sd, hyp_red, hyp_sd]"""

        return [dep_red, dep_sd]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['dep_red', 'dep_sd']
