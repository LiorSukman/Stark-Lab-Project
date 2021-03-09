import numpy as np
import math

class Graph(object):
    """
    An object used to represent a graph
    """
    def __init__(self, startNodes, endNodes, graphMatrix):
        self.startNodes = startNodes
        self.endNodes = endNodes
        self.graphMatrix = graphMatrix
        self.reversed = False
        self.edges = self._getAllEdges()

    def flip_graph(self):
        """
        Flips the values of the weights in the graph, i.e Positive weights will become negative
        """
        self.edges = [(e[0], e[1], e[2] * -1) for e in self.edges]
        self.reversed = not self.reversed

    def averageWeight(self):
        """
        returns:
        The average weight in the graph
        """
        total = 0
        counter = 0
        total = sum(e[2] for e in self.edges)
        counter = len(self.edges)

        if counter == 0:
            return 0
            
        return (total / counter)

    def _getAllEdges(self):
        """
        returns:
        a list of all edges in the graph. each edge is represented using a tupple:
            (from node, to node, weight of edge)
        """
        edges = []

        # Check to see if the grpah is currently reversed and determine what is the value that will
        # be used to indicate that an edge doesn't exist
        if self.reversed:
            not_exist = 1
        else:
            not_exist = -1

        for i in range(256):
            for j in range(256):
                if self.graphMatrix[i, j] != not_exist:
                    edges.append((i, j, self.graphMatrix[i, j]))
        return edges

    def _belmanFord(self, srcNode):
        """
        An implemetation of the Belman Ford algorithm to find shortest distances in a graph

        inputs:
        srcNode: The src node from which the search will begin

        returns:
        A list of size |nodes| containing the minimal distance from the source node to each other node
        """
        sizeV = 256 # this is |V|
        dists = [float('inf') for i in range(sizeV)]
        dists[srcNode] = 0

        edges = self.edges

        for i in range(sizeV - 1):
            for edge in edges:
                u = edge[0]
                v = edge[1]
                weight = edge[2]
                if dists[v] > dists[u] + weight:
                    dists[v] = dists[u] + weight

        return dists

    def _findMinimumDistToEndNodes(self, dists):
        """
        inputs:
        dists: the output of the Belman Ford algorithm

        returns:
        The minimum distnace to any of the end nodes of the graph
        """
        minimum = float('inf')
        for endNode in self.endNodes:
            if dists[endNode] < minimum:
                minimum = dists[endNode]
        return minimum

    def shortestDistanceFromSrcToEnd(self):
        """
        returns:
        The shortest distnace from any source node to any end node
        """
        totalMinDist = float('inf')
        for srcNode in self.startNodes:
            shortDists = self._belmanFord(srcNode)
            minDist = self._findMinimumDistToEndNodes(shortDists)
            if minDist < totalMinDist:
                totalMinDist = minDist
        return totalMinDist
    
    def longestDistanceFromSrcToEnd(self):
        """
        returns:
        The longest distnace from any source node to any end node
        """
        self.flip_graph()
        longest = self.shortestDistanceFromSrcToEnd()
        self.flip_graph()
        return longest * -1

class DepolarizationGraph(object):
    """
    This feature tries to estimate the way the signal traverses between the different channels. This traversal is modeled
    into a graph, where each node indicates a channel in a certain time, and each edge represents the speed in which the 
    signal travels between the two channels that comprise it.
    """
    def __init__(self):
        self.name = 'depolarization graph'

    def euclideanDist(self, pointA, pointB):
        """
        inputs:
        pointA: (x,y) tupple representing a point in 2D space
        pointB: (x,y) tupple representing a point in 2D space

        returns:
        The euclidean distance between the points
        """
        return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    def calculateDistancesMatrix(self, coordinates):
        """
        inputs:
        coordinates: a list of (x, y) tupples representing the coordinates of different channels

        returns:
        A 2D matrix in which a cell (i, j) contains the distance from coordiane i to coordinate j
        """
        distances = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                distances[i, j] = self.euclideanDist(coordinates[i], coordinates[j])

        return distances

    def get_indices_with_one(self, arr):
        """
        inputs:
        arr: a 2 dimensional matrix

        returns:
        the number of cells in the matrix that contain the value 1
        """
        lst = []
        for i in range(len(arr)):
            if arr[i] == 1:
                lst.append(i)
        return lst
    
    def calculateFeature(self, spikeList):
        """
        inputs:
        spikeList: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        # Determine the (x,y) coordinates of the 8 different channels and calculate the distances matrix
        coordinates = [(0, 0), (-9, 20), (8, 40), (-13, 60), (12, 80), (-17, 100), (16, 120), (-21, 140)]
        dists = self.calculateDistancesMatrix(coordinates)
        result = np.zeros((len(spikeList), 3))

        for index, spike in enumerate(spikeList):
            arr = spike.data
            minVal = arr.min()
            threshold = 0.3 * minVal # Setting the threshold to be 0.3 the size of max depolarization

            # Detrmine where the maximum depolarization resides wrt each channel (that surpasses the threshold)
            depolarizationStatus = np.zeros((8, 32))
            for i in range(8):
                maxDepIndex = arr[i].argmin()
                if arr[i, maxDepIndex] <= threshold:
                    depolarizationStatus[i, maxDepIndex] = 1

            # Find the channels that have reached max depolarization in each timestamp
            ds = depolarizationStatus
            gTemp = []
            for j in range(32):
                indices = self.get_indices_with_one(ds[:, j])
                if len(indices) > 0:
                    gTemp.append((j, indices))

            # Build the actual graph
            graphMatrix = np.ones((256, 256)) * (-1)
            startNodes = gTemp[0][1]
            endNodes = gTemp[len(gTemp)-1][1]
            for i in range(len(gTemp)-1):
                # each entry in gTemp is of the form (timestamp, list of indices)
                fromTimestamp = gTemp[i][0]
                for fromNode in gTemp[i][1]:
                    toTimestamp = gTemp[i+1][0]
                    for toNode in gTemp[i+1][1]:
                        #print(str(fromNode) + " -> " + str(toNode) + " With weight: " + str(distances[fromNode, toNode]))
                        velocity = dists[fromNode, toNode] / (toTimestamp - fromTimestamp)
                        graphMatrix[fromNode + fromTimestamp * 8][toNode + toTimestamp * 8] = velocity

            # Build the actual graph based on the data that was collected in the previous stage
            initialTime = gTemp[0][0] # Time of first channel reaching deplarization
            endTime = gTemp[len(gTemp)-1][0] # Time of last chaneel reaching depolarization            
            startNodes = [node + initialTime * 8 for node in startNodes] # The first nodes that reached depolarization
            endNodes = [node + endTime * 8 for node in endNodes] # The last nodes that reached depolarization
            graph = Graph(startNodes, endNodes, graphMatrix)
            
            # Calculate features from the graph
            result[index, 0] = graph.averageWeight()
            result[index, 1] = graph.shortestDistanceFromSrcToEnd()
            result[index, 2] = graph.longestDistanceFromSrcToEnd()

        return result

    def get_headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["graph_avg_speed", "graph_slowest_path", "graph_fastest_path"]

if __name__ == "__main__":
    f = DepolarizationGraph()
    mat = np.ones((8, 8)) * -1
    mat[0, 1] = 5
    mat[1, 2] = 2
    mat[0, 2] = 3
    G = Graph([0], [2], mat)
    print(G.shortestDistanceFromSrcToEnd())
    print(G.longestDistanceFromSrcToEnd())
