import numpy as np
import math


class Graph(object):
    """
    An object used to represent a graph
    """

    def __init__(self, start_nodes, end_nodes, graph_matrix):
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.graph_matrix = graph_matrix
        self.reversed = False
        self.edges = self._get_all_edges()

    def flip_graph(self):
        """
        Flips the values of the weights in the graph, i.e Positive weights will become negative
        """
        self.edges = [(e[0], e[1], e[2] * -1) for e in self.edges]
        self.reversed = not self.reversed

    def average_weight(self):
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

        return total / counter

    def _get_all_edges(self):
        """
        returns:
        a list of all edges in the graph. each edge is represented using a tupple:
            (from node, to node, weight of edge)
        """
        edges = []

        # Check to see if the graph is currently reversed and determine what is the value that will
        # be used to indicate that an edge doesn't exist
        if self.reversed:
            not_exist = 1
        else:
            not_exist = -1

        for i in range(256):
            for j in range(256):
                if self.graph_matrix[i, j] != not_exist:
                    edges.append((i, j, self.graph_matrix[i, j]))
        return edges

    def _bellman_ford(self, src_node):
        """
        An implementation of the Bellman Ford algorithm to find shortest distances in a graph

        inputs:
        src_node: The src node from which the search will begin

        returns:
        A list of size |nodes| containing the minimal distance from the source node to each other node
        """
        size_v = 256  # this is |V|
        dists = [float('inf') for i in range(size_v)]
        dists[src_node] = 0

        edges = self.edges

        for i in range(size_v - 1):
            for edge in edges:
                u = edge[0]
                v = edge[1]
                weight = edge[2]
                if dists[v] > dists[u] + weight:
                    dists[v] = dists[u] + weight

        return dists

    def _find_minimum_dist_to_end_nodes(self, dists):
        """
        inputs:
        dists: the output of the Bellman Ford algorithm

        returns:
        The minimum distance to any of the end nodes of the graph
        """
        minimum = float('inf')
        for end_node in self.end_nodes:
            if dists[end_node] < minimum:
                minimum = dists[end_node]
        return minimum

    def shortest_distance_from_src_to_end(self):
        """
        returns:
        The shortest distance from any source node to any end node
        """
        total_min_dist = float('inf')
        for src_node in self.start_nodes:
            short_dists = self._bellman_ford(src_node)
            min_dist = self._find_minimum_dist_to_end_nodes(short_dists)
            if min_dist < total_min_dist:
                total_min_dist = min_dist
        return total_min_dist

    def longest_distance_from_src_to_end(self):
        """
        returns:
        The longest distance from any source node to any end node
        """
        self.flip_graph()
        longest = self.shortest_distance_from_src_to_end()
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

    def euclidean_dist(self, point_a, point_b):
        """
        inputs:
        pointA: (x,y) tuple representing a point in 2D space
        pointB: (x,y) tuple representing a point in 2D space

        returns:
        The euclidean distance between the points
        """
        return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def calculate_distances_matrix(self, coordinates):
        """
        inputs:
        coordinates: a list of (x, y) tuples representing the coordinates of different channels

        returns:
        A 2D matrix in which a cell (i, j) contains the distance from coordinate i to coordinate j
        """
        distances = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                distances[i, j] = self.euclidean_dist(coordinates[i], coordinates[j])

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

    def calculate_feature(self, spike_lst):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        # Determine the (x,y) coordinates of the 8 different channels and calculate the distances matrix
        coordinates = [(0, 0), (-9, 20), (8, 40), (-13, 60), (12, 80), (-17, 100), (16, 120), (-21, 140)]
        dists = self.calculate_distances_matrix(coordinates)
        result = np.zeros((len(spike_lst), 3))

        for index, spike in enumerate(spike_lst):
            arr = spike.data
            min_val = arr.min()
            threshold = 0.3 * min_val  # Setting the threshold to be 0.3 the size of max depolarization

            # Determine where the maximum depolarization resides wrt each channel (that surpasses the threshold)
            depolarization_status = np.zeros((8, 32))
            for i in range(8):
                max_dep_index = arr[i].argmin()
                if arr[i, max_dep_index] <= threshold:
                    depolarization_status[i, max_dep_index] = 1

            # Find the channels that have reached max depolarization in each timestamp
            ds = depolarization_status
            g_temp = []
            for j in range(32):
                indices = self.get_indices_with_one(ds[:, j])
                if len(indices) > 0:
                    g_temp.append((j, indices))

            # Build the actual graph
            graph_matrix = np.ones((256, 256)) * (-1)
            start_nodes = g_temp[0][1]
            end_nodes = g_temp[len(g_temp) - 1][1]
            for i in range(len(g_temp) - 1):
                # each entry in g_temp is of the form (timestamp, list of indices)
                from_timestamp = g_temp[i][0]
                for fromNode in g_temp[i][1]:
                    to_timestamp = g_temp[i + 1][0]
                    for to_node in g_temp[i + 1][1]:
                        velocity = dists[fromNode, to_node] / (to_timestamp - from_timestamp)
                        graph_matrix[fromNode + from_timestamp * 8][to_node + to_timestamp * 8] = velocity

            # Build the actual graph based on the data that was collected in the previous stage
            initial_time = g_temp[0][0]  # Time of first channel reaching depolarization
            end_time = g_temp[len(g_temp) - 1][0]  # Time of last channel reaching depolarization
            # The first nodes that reached depolarization
            start_nodes = [node + initial_time * 8 for node in start_nodes]
            end_nodes = [node + end_time * 8 for node in end_nodes]  # The last nodes that reached depolarization
            graph = Graph(start_nodes, end_nodes, graph_matrix)

            # Calculate features from the graph
            result[index, 0] = graph.average_weight()
            result[index, 1] = graph.shortest_distance_from_src_to_end()
            result[index, 2] = graph.longest_distance_from_src_to_end()

        return result

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["graph_avg_speed", "graph_slowest_path", "graph_fastest_path"]


"""
if __name__ == "__main__":
    f = DepolarizationGraph()
    mat = np.ones((8, 8)) * -1
    mat[0, 1] = 5
    mat[1, 2] = 2
    mat[0, 2] = 3
    G = Graph([0], [2], mat)
    print(G.shortestDistanceFromSrcToEnd())
    print(G.longestDistanceFromSrcToEnd())
"""
