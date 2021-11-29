import numpy as np

from constants import NUM_CHANNELS

# There are two options of reduction for the da vector:
# ss - sum of squares
# sa - sum of absolutes
reduction_types = ['ss', 'sa']


# Direction_Agreeableness
class DA(object):
    """
    This feature estimates the amount of agreement in terms of directionality in relation to zero. 
    This feature pertains to the number of channels that are in agreement at each time sample, but not to the actual
    values of different channels (for this type of analysis see the ChannelContrastFeature).
    """

    def __init__(self, red_type='ss'):
        assert red_type in reduction_types
        self.red_type = red_type

        self.name = 'direction agreeableness feature'

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

    def calc_pos(self, arr, start_pos):
        left = start_pos
        while left >= 0:
            if arr[left] == 0:
                left -= 1
            else:
                break

        right = start_pos
        while right < len(arr):
            if arr[right] == 0:
                right += 1
            else:
                break

        return left, right

    def calc_feature_spike(self, spike):
        """
        inputs:
        spike: the spike to be processed; it is a matrix with the dimensions of (NUM_CHANNELS, TIMESTEPS * UPSAMPLE)

        The function calculates the direction agreeableness of the given spike

        returns:
        a tuple containing the sum of squares of the da vector and it's standard deviation
        """

        """median = np.median(spike)  # The median value in terms of amplitude across all channels
        direction = spike >= median
        counter = np.sum(direction, axis=0)

        # Iterating over the channels and calculating a direction agreeableness value
        for ind in range(counter.shape[0]):
            temp = counter[ind]
            counter[ind] = temp if temp <= NUM_CHANNELS // 2 else NUM_CHANNELS - temp

        # Reduce the da vector based on the chosen reduction type
        if self.red_type == 'ss':
            res = np.sum(counter ** 2)
        else:
            res = np.sum(counter)

        # Calculate the standard deviation of the da vector
        sd = np.std(counter)

        return [res, sd]"""

        med = np.median(spike)
        sig_m = np.where(spike <= med, -1, 1)
        changes = np.zeros((2, len(sig_m)))
        for i, channel in enumerate(sig_m):
            change = np.convolve(channel, [-1, 1])
            l, r = self.calc_pos(change, spike[i].argmin())
            changes[0, i] = l
            changes[1, i] = r

        return changes.std(axis=1).flatten()

    @property
    def headers(self):
        #return ['da', 'da_sd']
        return ['da_left', 'da_right']
