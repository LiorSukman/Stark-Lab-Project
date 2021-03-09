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

    def euclideanDist(self, pointA, pointB):
        """
        inputs:
        pointA: (x,y) tupple representing a point in 2D space
        pointB: (x,y) tupple representing a point in 2D space

        returns:
        The euclidean distance between the points
        """
        return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    def calculate_geo_estimation(self, channelsAtTime, coordinates):
        """
        inputs:
        channelsAtTime: a list of the value that was samples across all channels at a certain time
        coordinates: a list of (x, y) tupples representing the location of the different channels on a 2D plane

        returns:
        A tupple containing the X and Y coordinates of the geometrical estimation
        """
        maxVal = channelsAtTime.max()
        
        total = 0
        for i in range(8):
            entry = channelsAtTime[i]
            if entry < 0:
                entry *= -1
            total += entry
        channelsAtTime = channelsAtTime / total
        geoX = sum([coordinates[i][0] * channelsAtTime[i] for i in range(8)]) # Estimation for the X coordinate
        geoY = sum([coordinates[i][1] * channelsAtTime[i] for i in range(8)]) # Estimation for the Y coordinate
        return (geoX, geoY)

    def calculate_shifts_2D(self, geoAvgs):
        """
        inputs:
        geoAvgs: a list of geometrical averages (each with an X and Y coordinates)

        returns:
        a vecotr of dimensions (1, 31) where entry i represnts the shift in terms of euclidean distance between
            the geometrical estimation between sample i and sample i-1
        """
        shifts = np.zeros((1, 31))
        for i in range(1, 32):
            shifts[0][i - 1] = self.euclideanDist((geoAvgs[i - 1][0], geoAvgs[i-1][1]), (geoAvgs[i][0], geoAvgs[i][1]))
        return shifts

    def calculate_shifts_1D(self, geoAvgs, d):
        """
        inputs:
        geoAvgs: a list of geometrical averages (each with an X and Y coordinates)
        d: the dimension that will be included in the calculation (0 or 1 - X or Y)

        returns:
        a vecotr of dimensions (1, 31) where entry i represnts the shift in terms of euclidean distance between
            one of the dimensions of the geometrical estimation between sample i and sample i-1
        """
        shifts = np.zeros((1, 31))
        for i in range(1, 32):
            shifts[0][i - 1] = geoAvgs[i][d] - geoAvgs[i-1][d]
        return shifts

    def calc_max_dist(self, cordinates):
        """
        inputs:
        coordinates: a list of (x, y) tupples representing the location of geometrical estimations

        returns:
        the max distance between two coordinates
        """
        max_dist = 0
        for i, cor1 in enumerate(cordinates[:-1]):
            for cor2 in cordinates[i + 1:]:
                dist = self.euclideanDist((cor1[0], cor1[1]), (cor2[0], cor2[1]))
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def calculateFeature(self, spikeList):
        """
        inputs:
        spikeList: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = np.zeros((len(spikeList), 3))
        coordinates = [(0, 0), (-9, 20), (8, 40), (-13, 60), (12, 80), (-17, 100), (16, 120), (-21, 140)]

        for j, spike in enumerate(spikeList):
            geoAvgs = np.zeros((32, 2))

            arr = spike.get_data()
            for i in range(32):
                channels = arr[:, i] * (-1) # channels that are positive need to be considered in reverse in terms of average claculation
                geoAvgs[i, 0], geoAvgs[i, 1] = self.calculate_geo_estimation(channels, coordinates)

            shifts_2D = self.calculate_shifts_2D(geoAvgs)

            result[j, 0] = np.mean(shifts_2D, axis = 1)
            result[j, 1] = np.std(shifts_2D, axis = 1)
            #result[j, 2] = self.euclideanDist((geoAvgs[0][0], geoAvgs[0][1]), (geoAvgs[-1][0], geoAvgs[-1][1]))
            result[j, 2] = self.calc_max_dist(geoAvgs)
        return result

    def get_headers(self):
        """
        Returns a list of titles of the different metrics
        """
        #return ["geometrical_avg_shift", "geometrical_shift_sd", "geometrical_displacement", "geometrical_max_dist"]
        return ["geometrical_avg_shift", "geometrical_shift_sd", "geometrical_max_dist"]
