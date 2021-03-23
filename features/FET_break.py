import numpy as np


# TODO fix all descriptions, maybe change the name of the file
# TODO consider all parameters, d&c had different values in the report and the file
# TODO the feature was calculated on the normalized spike (divided by sum of squares), should I?

def calc_second_der(spike):
    # TODO consider dividing by something of the time
    # TODO consider using np.gradient
    first_der = np.convolve(spike, [1, -1], mode='valid')
    second_der = np.convolve(first_der, [1, -1], mode='valid')

    # should be equivalent to:
    # der = np.convolve(spike, [1, -2, 1], mode='valid')

    return second_der


class BreakMeasurement(object):
    """
    TODO add description
    """

    def __init__(self, start=-7, end=-3, eps=0, mul_const=20):
        self.start = start
        self.end = end
        self.eps = eps
        self.mul_const = mul_const

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
        der = calc_second_der(spike)
        roi = der[dep_ind + self.start: dep_ind + self.end]

        ret = self.mul_const * np.log(np.exp(np.sum(roi)) - 1 + self.eps)

        return ret

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['break_measure']
