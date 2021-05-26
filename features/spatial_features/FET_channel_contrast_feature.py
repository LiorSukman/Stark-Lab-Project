import numpy as np

from constants import NUM_CHANNELS


def find_dominant_channel(spike):
    """
    inputs:
    spike: the spike to be processed; it is a matrix with the dimensions of (NUM_CHANNELS, TIMESTEPS)

    returns:
    the channel that contains the maximum depolarization and the sample in which that depolarization occurs
    """
    arg_min_channel = spike.min(axis=1).argmin()
    arg_min_time = spike.min(axis=0).argmin()
    return arg_min_channel, arg_min_time


class ChannelContrast(object):
    """
    This feature estimates the actual agreement between different channels in relation to zero. While the DA Feature
    only deals in the absolute number of channels that are in disagreement, this feature expands on that and aspires to
    actually model the pattern of disagrement using dot products between dhifferent channels and the main channel.
    """

    def __init__(self):
        self.name = 'channel contrast feature'

    def calculate_feature(self, spike_lst):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = np.zeros((len(spike_lst), 1))
        for i, spike in enumerate(spike_lst):

            # Find the dominant channel
            dominant_channel, dom_time = find_dominant_channel(spike.data)
            reduced_arr = spike.data / spike.data.shape[1]

            # Iterate over the other channels and check the contrast wrt the dominant one
            res = np.zeros((1, NUM_CHANNELS))
            for j in range(NUM_CHANNELS):
                if j != dominant_channel:
                    """
                    TODO: consider just looking at the nonagreeing steps (i.e. instead of
                    reduced_arr[dominant_channel] use reduced_arr[dominant_channel] *
                    ((reduced_arr[j] * reduced_arr[dominant_channel]) < 0)
                    """
                    dot = np.dot(reduced_arr[j], reduced_arr[dominant_channel])

                    # if there is a contrast write it, o.w write zero
                    res[0, j] = dot if dot < 0 else 0

            result[i, 0] = np.sum(res * -1)
        return result

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["Channels contrast"]
