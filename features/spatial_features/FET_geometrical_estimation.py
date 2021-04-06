import numpy as np
import math
from scipy.spatial.distance import cdist

from constants import NUM_CHANNELS, TIMESTEPS, UPSAMPLE, COORDINATES

class GeometricalEstimation(object):
    """
    This feature estimates the geometrical location of the signal at each time sample and gives and perofrms various
    calculations that are based on these estimates.
    These calculations pertain to the change in location of the geometrical estimations.
    """

    def __init__(self):
        self.name = 'Geometrical estimation'

    def euclidean_dist(self, point_a, point_b):
        """
        inputs:
        pointA: (x,y) tuple representing a point in 2D space
        pointB: (x,y) tuple representing a point in 2D space

        returns:
        The euclidean distance between the points
        """
        return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def calculate_geo_estimation(self, channels_at_time, coordinates):
        """
        inputs:
        channels_at_time: a list of the value that was samples across all channels at a certain time
        coordinates: a list of (x, y) tuples representing the location of the different channels on a 2D plane

        returns:
        (geo_x, geo_y): A tuple containing the X and Y coordinates of the geometrical estimation
        """
        total = np.absolute(channels_at_time).sum()
        channels_at_time = channels_at_time / total
        # Estimation for the X coordinate
        geo_x = np.dot(coordinates[:, 0], channels_at_time)
        # Estimation for the Y coordinate
        geo_y = np.dot(coordinates[:, 1], channels_at_time)
        return geo_x, geo_y

    def calculate_shifts_2d(self, dists):
        """
        inputs:
        geo_avgs: a matrix of distances between all pairs of geometrical averages

        returns:
        a vector of dimensions (1, TIMESTEPS * UPSAMPLE - 1) where entry i represents the shift in terms of euclidean
        distance between the geometrical estimation between sample i and sample i-1
        """
        shifts = np.zeros((1, TIMESTEPS * UPSAMPLE - 1))
        for i in range(1, TIMESTEPS * UPSAMPLE):
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

    def calc_max_dist(self, coordinates):
        """
        inputs:
        coordinates: a list of (x, y) tuples representing the location of geometrical estimations

        returns:
        the max distance between two coordinates
        """
        max_dist = 0
        for i, cor1 in enumerate(coordinates[:-1]):
            for cor2 in coordinates[i + 1:]:
                dist = self.euclidean_dist((cor1[0], cor1[1]), (cor2[0], cor2[1]))
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def calculate_feature(self, spike_lst):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = np.zeros((len(spike_lst), 3))
        coordinates = COORDINATES

        for j, spike in enumerate(spike_lst):
            geo_avgs = np.zeros((TIMESTEPS * UPSAMPLE, 2))

            arr = spike.get_data()
            for i in range(TIMESTEPS * UPSAMPLE):
                # channels that are positive need to be considered in reverse in terms of average calculation
                channels = arr[:, i] * (-1)
                geo_avgs[i, 0], geo_avgs[i, 1] = self.calculate_geo_estimation(channels, coordinates)

            dists = cdist(geo_avgs, geo_avgs)

            shifts_2d = self.calculate_shifts_2d(dists)

            result[j, 0] = np.mean(shifts_2d, axis=1)
            result[j, 1] = np.std(shifts_2d, axis=1)
            result[j, 2] = dists.max()
        return result

    @ property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        # return ["geometrical_avg_shift", "geometrical_shift_sd", "geometrical_displacement", "geometrical_max_dist"]
        return ["geometrical_avg_shift", "geometrical_shift_sd", "geometrical_max_dist"]
