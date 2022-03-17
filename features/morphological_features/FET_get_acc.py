import numpy as np
from constants import INF

# TODO consider all parameters, d&c had different values in the report and the file
# TODO the feature was calculated on the normalized spike (divided by sum of squares), should I?

def calc_second_der(spike):
    """
    inputs:
    spike: the spike to be processed; it is an ndarray with TIMESTEPS * UPSAMPLE entries

    returns:
    The second order derivative of the spike calculated according to y'(t)=y(t+1)-y(t)
    """
    # TODO: this is the same as in FET_smile_cry and FET_break, might want to create util
    # TODO consider dividing by something of the time
    # TODO consider using np.gradient
    first_der = np.convolve(spike, [1, -1], mode='valid')
    second_der = np.convolve(first_der, [1, -1], mode='valid')

    # should be equivalent to:
    # der = np.convolve(spike, [1, -2, 1], mode='valid')

    return second_der


class GetAcc(object):
    """
    This feature aims to quantify the rate in which the spike decreases its voltage gaining rate quickly before
    returning to a somewhat steady rate, that corresponds to a second derivative close to zero.
    """

    def __init__(self, start=14, end=40, mul_const=20):
        # the start and end constants correspond to 0.3ms to 0.085ms before depolarization based on a sampling rate of
        # 20kHz and an upsampling by a factor of 8.
        self.start = start  # start of the region of interest in relation to the depolarization
        self.end = end  # end of the region of interest in relation to the depolarization
        # constants used for the calculation of the final value
        self.mul_const = mul_const

        self.name = 'get acc'

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
        spike: the spike to be processed; it is an ndarray with TIMESTEPS * UPSAMPLE entries

        The function calculates the get acc feature as described above

        returns: a list containing the get acc feature
        """
        dep_ind = np.argmin(spike)
        der = calc_second_der(spike)
        roi = der[dep_ind + self.start: dep_ind + self.end]

        # assert np.sum(roi ** 2) > 0 the following if statement was used to handle this case yet it is not necessarily
        # a good solution, uncomment this to find more problematic spikes

        if len(roi) == 0:
            return [-INF]

        ret = np.sum(roi ** 2)  # TODO: reconsider multiplicative constants

        return [ret]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['get_acc']
