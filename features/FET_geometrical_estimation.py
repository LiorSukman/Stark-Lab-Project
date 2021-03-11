import numpy as np
import math


class GeometricalEstimation(object):
    """
    This feature estimates the geometrical location of the signal at each time sample and gives and perofrms various calculations 
    that are based on these estimates.
    These calculations pertain to the change in location of the geometrical estimations.
    """

    def __init__(self):
        self.name = 'Geometrical estimation'

    def euclidean_dist(self, point_a, point_b):
        """
        inputs:
        pointA: (x,y) tupple representing a point in 2D space
        pointB: (x,y) tupple representing a point in 2D space

        returns:
        The euclidean distance between the points
        """
        return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def calculate_geo_estimation(self, channels_at_time, coordinates):
        """
        inputs:
        channels_at_time: a list of the value that was samples across all channels at a certain time
        coordinates: a list of (x, y) tupples representing the location of the different channels on a 2D plane

        returns:
        (geo_x, geo_y): A tuple containing the X and Y coordinates of the geometrical estimation
        """
        # TODO check why it is not used
        max_val = channels_at_time.max()

        total = 0
        for i in range(8):
            entry = channels_at_time[i]
            if entry < 0:
                entry *= -1
            total += entry
        channels_at_time = channels_at_time / total
        geo_x = sum([coordinates[i][0] * channels_at_time[i] for i in range(8)])  # Estimation for the X coordinate
        geo_y = sum([coordinates[i][1] * channels_at_time[i] for i in range(8)])  # Estimation for the Y coordinate
        return geo_x, geo_y

    def calculate_shifts_2d(self, geo_avgs):
        """
        inputs:
        geo_avgs: a list of geometrical averages (each with an X and Y coordinates)

        returns:
        a vector of dimensions (1, 31) where entry i represnts the shift in terms of euclidean distance between
            the geometrical estimation between sample i and sample i-1
        """
        shifts = np.zeros((1, 31))
        for i in range(1, 32):
            shifts[0][i - 1] = self.euclidean_dist((geo_avgs[i - 1][0], geo_avgs[i - 1][1]),
                                                   (geo_avgs[i][0], geo_avgs[i][1]))
        return shifts

    def calculate_shifts_1d(self, geo_avgs, d):
        """
        inputs:
        geo_avgs: a list of geometrical averages (each with an X and Y coordinates)
        d: the dimension that will be included in the calculation (0 or 1 - X or Y)

        returns:
        a vector of dimensions (1, 31) where entry i represents the shift in terms of euclidean distance between
            one of the dimensions of the geometrical estimation between sample i and sample i-1
        """
        shifts = np.zeros((1, 31))
        for i in range(1, 32):
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
        coordinates = [(0, 0), (-9, 20), (8, 40), (-13, 60), (12, 80), (-17, 100), (16, 120), (-21, 140)]

        for j, spike in enumerate(spike_lst):
            geo_avgs = np.zeros((32, 2))

            arr = spike.get_data()
            for i in range(32):
                channels = arr[:, i] * (
                    -1)  # channels that are positive need to be considered in reverse in terms of average claculation
                geo_avgs[i, 0], geo_avgs[i, 1] = self.calculate_geo_estimation(channels, coordinates)

            shifts_2d = self.calculate_shifts_2d(geo_avgs)

            result[j, 0] = np.mean(shifts_2d, axis=1)
            result[j, 1] = np.std(shifts_2d, axis=1)
            # result[j, 2] = self.euclideanDist((geo_avgs[0][0], geo_avgs[0][1]), (geo_avgs[-1][0], geo_avgs[-1][1]))
            result[j, 2] = self.calc_max_dist(geo_avgs)
        return result

    @ property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        # return ["geometrical_avg_shift", "geometrical_shift_sd", "geometrical_displacement", "geometrical_max_dist"]
        return ["geometrical_avg_shift", "geometrical_shift_sd", "geometrical_max_dist"]
