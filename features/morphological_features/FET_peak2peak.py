import numpy as np


# TODO fix all descriptions, maybe change the name of the file

class Peak2Peak(object):
    """
    TODO add description
    """

    def __init__(self):
        pass

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
        spike: the spike to be processed; it is a matrix with the dimensions of (8, 32)

        The function calculates...

        returns: a list containing...
        """
        # find timestamps for depolarization in ok channels, filter again to assure depolarization is reached before the
        # end
        dep_ind = np.argmin(spike)
        dep = spike[dep_ind]
        if dep_ind == len(spike):  # if max depolarization is reached at the end, it indicates noise
            raise Exception('Max depolarization reached at final timestamp')

        trun_spike = spike[dep_ind + 1:]
        hyp_ind = trun_spike.argmax() + dep_ind + 1
        hyp = spike[hyp_ind]

        peak_2_peak = hyp - dep
        trough_2_peak = hyp_ind - dep_ind

        return peak_2_peak, trough_2_peak

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['peak2peak', 'trough2peak']
