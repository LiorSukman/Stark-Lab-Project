import numpy as np
import matplotlib.pyplot as plt


class RiseCoef(object):
    """
    This feature finds the most distant point in the spike from the linear line connecting the depolarization with the
    final value of the recording (only looking from the depolarization and on)
    """

    def __init__(self):
        self.name = 'rise_coefficient'

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

        The function calculates the rise coefficient as described above.

        returns: a list containing the value of the rise coefficient
        """
        dep_ind = np.argmin(spike)
        dep = spike[dep_ind]
        """if dep_ind == len(spike) - 1:  # if max depolarization is reached at the end, it indicates noise
            plt.plot(spike)
            plt.show()
            raise Exception('Max depolarization reached at final timestamp')"""
        line = np.linspace(dep, spike[-1], num=len(spike) - dep_ind)

        trun_spike = spike[dep_ind:]
        rise_coef = (trun_spike - line).argmax()

        return [rise_coef]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ['rise_coef']
