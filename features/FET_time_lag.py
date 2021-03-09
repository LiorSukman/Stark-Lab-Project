import numpy as np
from clusters import Spike

# There are two options of reduction for the da vector:
# ss - sum of squares
# sa - sum of absolutes
reduction_types = ['ss', 'sa']

class Time_Lag_Feature(object):
    """
    This feature calculates the time difference between the main channel and all other channels in terms of
    maximal depolarization, and the following after hyperpolarization.
    The feature only takes into consideration channels that have crossed a ceratin threshold, dtermined by the 
    maximal depolarization of the main channel.
    """
    def __init__(self, type_dep = 'ss', type_hyp = 'ss', ratio = 0.25):
        assert type_dep in reduction_types and type_hyp in reduction_types
        
        # Reduction types
        self.type_dep = type_dep
        self.type_hyp = type_hyp

        self.ratio = ratio # Indicates the percantage of the maximum depolarization that will be considered as a threshold

        self.name = 'time lag feature'

    def calculateFeature(self, spikeList):
        """
        inputs:
        spikeList: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = [self.calc_feature_spike(spike.get_data()) for spike in spikeList]
        result = np.asarray(result)
        return result

    def calc_feature_spike(self, spike):
        """
        inputs:
        spike: the spike to be processed; it is a matrix with the dimensions of (8, 32)

        The function calculates different time lag features of the spike

        returns:
        a list containing the following values:
            -dep_red: the reduction of the depolarization vactor (i.e the vector that indicates the time difference of maximal depolarization between each channel and the main channel)
            -dep_sd: the standard deviation of the depolarization vector
            -hyp_red: the reduction of the hyperpolarization vector
            -hyp_sd: the standard deviation of the hyperpolarization vector
        """
        # remove channels with lower depolarization than required
        deps = np.min(spike, axis = 1) # max depolarization of each channel
        max_dep = np.min(deps)
        fix_inds = deps <= self.ratio * max_dep
        dep_ind = np.argmin(spike, axis = 1)
        spike = spike[fix_inds]

        # find timesteps for depolarizrion in ok chanells, filter again to assure depolariztion is reached before the end
        dep_ind = np.argmin(spike, axis = 1)
        fix_inds = dep_ind < 31 # if max depolariztion is reached at the end, it indicates noise
        dep_ind = dep_ind[fix_inds]
        spike = spike[fix_inds]
        if spike.shape[0] == 0: # if no channel passes filtering return zeros
            return [0, 0, 0, 0]

        # offset according to the main channel
        main_chn = np.argmin(spike) // 32 # set main channel to be the one with highest depolariztion
        dep_rel = dep_ind - dep_ind[main_chn] # offsetting

        # calculate sd of depolarization time differences
        dep_sd = np.std(dep_rel)

        # calculate reduction
        if self.type_dep == 'ss':
            dep_red = np.sum(dep_rel ** 2)
        else: #i.e sa
            dep_red = np.sum(np.absolute(dep_rel))

        # find hyperpolarization indeces
        hyp_ind = []
        for i, channel in enumerate(spike):
            trun_channel = channel[dep_ind[i] + 1:]
            hyp_ind.append(trun_channel.argmax() + dep_ind[i] + 1)
        hyp_ind = np.asarray(hyp_ind)

        # repeat calulations                 
        hyp_rel = hyp_ind - hyp_ind[main_chn]
        hyp_sd = np.std(hyp_rel)
        if self.type_hyp == 'ss':
            hyp_red = np.sum(hyp_rel ** 2)
        else: #i.e sa
            hyp_red = np.sum(np.absolute(hyp_rel))
        
        return [dep_red, dep_sd, hyp_red, hyp_sd]

    def get_headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['dep_red', 'dep_sd', 'hyp_red', 'hyp_sd']

    
