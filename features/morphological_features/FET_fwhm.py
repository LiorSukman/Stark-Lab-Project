import numpy as np


class FWHM(object):
    """
    This feature times the duration in which the recording shows some ratio of the maximal depolarization (0.5 at
    default). This gives a sense of the length of the spike.
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio  # the threshold ratio of the max depolarization

        self.name = 'FWHM'

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

        The function calculates the fwhm value as described above.

        returns: a list containing the fwh mvalue
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
