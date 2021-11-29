import numpy as np
import math
from scipy.spatial.distance import cdist

from constants import TIMESTEPS, UPSAMPLE, COORDINATES

class GeometricalEstimation(object):
    """
    This feature estimates the geometrical location of the signal at each time sample and gives and perofrms various
    calculations that are based on these estimates.
    These calculations pertain to the change in location of the geometrical estimations.
    """

    def __init__(self):
        self.name = 'Geometrical estimation'

    def remove_not_valid(self, geo_avgs, is_valid):
        start, end = 0, 0
        for i in range(len(geo_avgs)):
            if is_valid[i]:
                start = i
                break

        for i in range(len(geo_avgs) - 1, -1, -1):
            if is_valid[i]:
                end = i + 1
                break

        return geo_avgs[start: end]

    def calculate_geo_estimation(self, channels_at_time, coordinates):
        """
        inputs:
        channels_at_time: a list of the value that was samples across all channels at a certain time
        coordinates: a list of (x, y) tuples representing the location of the different channels on a 2D plane

        returns:
        (geo_x, geo_y): A tuple containing the X and Y coordinates of the geometrical estimation
        """
        total = np.absolute(channels_at_time).sum()
        if total == 0:
            geo_x, geo_y = coordinates.mean(axis=0)
            return geo_x, geo_y, False
        channels_at_time = channels_at_time / total
        # Estimation for the X coordinate
        geo_x = np.dot(coordinates[:, 0], channels_at_time)
        # Estimation for the Y coordinate
        geo_y = np.dot(coordinates[:, 1], channels_at_time)
        return geo_x, geo_y, True

    def calculate_shifts_2d(self, dists):
        """
        inputs:
        geo_avgs: a matrix of distances between all pairs of geometrical averages

        returns:
        a vector of dimensions (1, TIMESTEPS * UPSAMPLE - 1) where entry i represents the shift in terms of euclidean
        distance between the geometrical estimation between sample i and sample i-1
        """
        shifts = np.zeros((1, len(dists)))
        for i in range(1, len(dists)):
            shifts[0][i - 1] = dists[i-1, i]
        return shifts

    def calculate_shifts_1d(self, geo_avgs, d):
        """
        inputs:
        geo_avgs: a list of geometrical averages (each with an X and Y coordinates)
        d: the dimension that will be included in the calculation (0 or 1 - X or Y)

        returns:
        a vector of dimensions (1, TIMESTEPS * UPSAMPLE - 1) where entry i represents the shift in terms of euclidean
         distance between one of the dimensions of the geometrical estimation between sample i and sample i-1
        """
        shifts = np.zeros((1, TIMESTEPS * UPSAMPLE - 1))
        for i in range(1, TIMESTEPS * UPSAMPLE):
            shifts[0][i - 1] = geo_avgs[i][d] - geo_avgs[i - 1][d]
        return shifts

    def calculate_feature(self, wvlt_lst):
        """
        inputs:
        wvlt_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        # result = np.zeros((len(wvlt_lst), 3))
        result = np.zeros((len(wvlt_lst), 2))
        coordinates = COORDINATES

        for j, spike in enumerate(wvlt_lst):
            geo_avgs = np.zeros((TIMESTEPS * UPSAMPLE, 2))
            is_valid = np.zeros((TIMESTEPS * UPSAMPLE))

            arr = spike.get_data()
            for i in range(TIMESTEPS * UPSAMPLE):
                # channels that are positive need to be considered in reverse in terms of average calculation
                channels = arr[:, i] * (-1)
                x, y, valid = self.calculate_geo_estimation(channels, coordinates)
                geo_avgs[i, 0], geo_avgs[i, 1] = x, y
                is_valid[i] = valid

            geo_avgs = self.remove_not_valid(geo_avgs, is_valid)
            dists = cdist(geo_avgs, geo_avgs)
            shifts_2d = self.calculate_shifts_2d(dists)

            result[j, 0] = np.mean(shifts_2d, axis=1)
            result[j, 1] = np.std(shifts_2d, axis=1)

        return result

    @ property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["geometrical_shift", "geometrical_shift_sd"]
