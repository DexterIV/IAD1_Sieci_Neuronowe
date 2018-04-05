from DataStructure.Cluster import Cluster
from DataStructure.Centroid import Centroid
from DataStructure.Data import Data
from UniversalFunctions import *
import pandas as pd


class KMeans:
    def __init__(self, number_of_clusters=3, max_iterations=64, absolute_tolerance=0.000001,
                 little_data_threshold=0.015):
        self.clusters = []
        self.numberOfClusters = number_of_clusters
        self.maxIterations = max_iterations
        self.absoluteTolerance = absolute_tolerance
        self.data = []
        self.littleDataThreshold = little_data_threshold

    def initialize_data(self, filename):
        self.data.clear ()
        dataset = pd.read_csv (filename)
        number_of_columns = len (dataset.columns)
        data_attributes = []

        for i in range (1, number_of_columns - 1):
            data_attributes.append (dataset.iloc[:, i].values)

        for i in range (len (data_attributes[0])):
            values = []
            for j in range (len (data_attributes)):
                values.append (data_attributes[j][i])
            self.data.append (Data (values))

    def initialize_centroids(self):
        for i in range(self.numberOfClusters):
            import random
            centroids_position = Data(self.data[random.randint(0, len(self.data) - 1)].values)
            centroid = Centroid(centroids_position)
            self.clusters.append(Cluster(centroid, True))

    def algorithm(self):
        i = 0
        for i in range(self.maxIterations):
            clear_clusters(self.clusters)
            self._assign_data_to_clusters()
            self._move_centroids()
            self._reassign_clusters_with_little_data()
            if self._second_stop_condition():
                break
        plot_all_clusters(self.clusters, i, 'KMeans algorithm', len(self.data[0].values), None)

    def _assign_data_to_clusters(self):
        for j in range(len(self.data)):
            closest_centroid_index = self._find_minimum_distance(self.data[j])
            self.clusters[closest_centroid_index].data.append(self.data[j])

    def _move_centroids(self):
        for i in range(len(self.clusters)):
            copy_values(self.clusters[i].centroid.position.values,
                        self.clusters[i].previous_centroid.position.values)
            new_centroid_position = []
            new_centroid_position.clear()
            new_centroid_position = [0] * len(self.data[0].values)
            for j in range(len(new_centroid_position)):
                for k in range(len(self.clusters[i].data)):
                    new_centroid_position[j] += self.clusters[i].data[k].values[j]
                new_centroid_position[j] /= len(self.clusters[i].data)
            copy_values(new_centroid_position, self.clusters[i].centroid.position.values)

    def _reassign_clusters_with_little_data(self):
        import random
        for i in range(self.numberOfClusters):
            if len(self.clusters[i].data) <= self.littleDataThreshold * len(self.data):
                self.clusters[i].centroid.position = self.data[random.randint(0, len(self.data) - 1)]

    def _find_minimum_distance(self, data_instance):
        index = 0
        minimum = distance(data_instance.values, self.clusters[index].centroid.position.values)
        for i in range(1, len(self.clusters)):
            dist = distance(self.clusters[i].centroid.position.values, data_instance.values)
            if dist < minimum:
                minimum = dist
                index = i
        return index

    def _second_stop_condition(self):
        result = True
        for i in range(self.numberOfClusters):
            if distance(self.clusters[i].centroid.position.values,
                        self.clusters[i].previous_centroid.position.values) > self.absoluteTolerance:
                result = False
        return result
