import numpy as np


# TODO fix all descriptions, maybe change the name of the file

class RiseCoef(object):
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
        dep_ind = np.argmin(spike)
        dep = spike[dep_ind]
        if dep_ind == len(spike):  # if max depolarization is reached at the end, it indicates noise
            raise Exception('Max depolarization reached at final timestamp')
        line = np.linspace(dep, spike[-1], num = len(spike) - dep_ind + 1)  # make sure the spike is actually a single dimensional array

        trun_spike = spike[dep_ind:]
        rise_coef = (trun_spike - line).argmax()

        return rise_coef

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['rise_coef']
