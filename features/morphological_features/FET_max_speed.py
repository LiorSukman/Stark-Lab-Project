import numpy as np


# TODO fix all descriptions, maybe change the name of the file
# TODO the feature was calculated on the normalized spike (divided by sum of squares), should I?

def calc_der(spike):
    # TODO consider dividing by something of the time
    # TODO consider using np.gradient
    first_der = np.convolve(spike, [1, -1], mode='valid')

    return first_der


class MaxSpeed(object):
    """
    TODO add description
    """

    def __init__(self, start=131):
        self.start = start

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
        der = calc_der(spike)
        der_roi = der[self.start:]

        ret = der_roi[der_roi < der_roi[0]].argmax()

        return ret

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['max_speed']
