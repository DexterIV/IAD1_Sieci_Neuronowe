from DataStructure.Cluster import Cluster
from DataStructure.Centroid import Centroid
from DataStructure.Data import Data
from UniversalFunctions import *
import pandas as pd
import numpy


class KMeans:
    def __init__(self, number_of_clusters=3, show_plots=True, max_iterations=64, absolute_tolerance=0.000001,
                 little_data_threshold=0.015):
        self.clusters = []
        self.numberOfClusters = number_of_clusters
        self.maxIterations = max_iterations
        self.absoluteTolerance = absolute_tolerance
        self.data = []
        self.littleDataThreshold = little_data_threshold
        self.show_plots = show_plots
        self.dataLabels = []

    def initialize_data(self, filename):
        self.data.clear()
        dataset_tmp = pd.read_csv(filename, header=None)
        number_of_columns = len(dataset_tmp.columns)
        data_attributes = []

        for i in range(1, number_of_columns - 1):
            self.dataLabels.append(dataset_tmp[i][0])

        dataset = pd.read_csv(filename)

        for i in range(1, number_of_columns - 1):
            data_attributes.append(dataset.iloc[:, i].values)

        for i in range(1, len(data_attributes[0])):
            values = []
            for j in range(len(data_attributes)):
                values.append(data_attributes[j][i])
            self.data.append(Data(values))

    def initialize_data_with_chunks(self, chunks):
        for i in range(len(chunks)):
            self.data.append(Data(convert_3d_to_1d_list(chunks[i])))

    def initialize_centroids(self):
        for i in range(self.numberOfClusters):
            import random
            centroids_position = Data(self.data[random.randint(0, len(self.data) - 1)].values)
            centroid = Centroid(centroids_position)
            self.clusters.append(Cluster(centroid, True))

    def algorithm(self):
        i = 0
        for i in range(self.maxIterations):
            self._clear_clusters(self.clusters)
            self._assign_data_to_clusters()
            self._move_centroids()
            if i % 8 == 0:
                self._reassign_clusters_with_little_data()
            print(i)
            if self._second_stop_condition():
                break
        if self.show_plots:
            plot_all_clusters(self.clusters, i, 'KMeans algorithm', len(self.data[0].values), None, self.dataLabels)
        self._average_distance_from_centroid()
        return self.clusters

    def _assign_data_to_clusters(self):
        for j in range(len(self.data)):
            closest_centroid_index = self._find_minimum_distance(self.data[j])
            self.clusters[closest_centroid_index].data.append(self.data[j])

    def _move_centroids(self):
        for i in range(len(self.clusters)):
            self.clusters[i].previous_centroid.position.values = numpy.copy(self.clusters[i].centroid.position.values)
            new_centroid_position = []
            new_centroid_position.clear()
            new_centroid_position = [0] * len(self.data[0].values)
            for j in range(len(new_centroid_position)):
                if len(self.clusters[i].data) != 0:
                    for k in range(len(self.clusters[i].data)):
                        new_centroid_position[j] += self.clusters[i].data[k].values[j]
                    new_centroid_position[j] /= len(self.clusters[i].data)
                else:
                    import random
                    self.clusters[i].centroid.position = self.data[random.randint(0, len (self.data) - 1)]
            self.clusters[i].centroid.position.values = numpy.copy(new_centroid_position)

    def _reassign_clusters_with_little_data(self):
        import random
        for i in range(self.numberOfClusters):
            if len(self.clusters[i].data) <= self.littleDataThreshold * len(self.data):
                self.clusters[i].centroid.position = self.data[random.randint(0, len(self.data) - 1)]

    def _find_minimum_distance(self, data_instance):
        index = 0
        minimum = distance_for_comparison(data_instance.values, self.clusters[index].centroid.position.values)
        for i in range(1, len(self.clusters)):
            dist = distance_for_comparison(self.clusters[i].centroid.position.values, data_instance.values)
            if dist < minimum:
                minimum = dist
                index = i
        return index

    def _second_stop_condition(self):
        result = True
        for i in range(self.numberOfClusters):
            dist = distance_for_comparison(self.clusters[i].centroid.position.values,
                                           self.clusters[i].previous_centroid.position.values)
            if dist > self.absoluteTolerance:
                result = False
        return result

    def _clear_clusters(self):
        for i in range(len(self.clusters)):
            self.clusters[i].data.clear()

    def _average_distance_from_centroid(self):
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i].data)):
                self.clusters[i].error += distance(self.clusters[i].data[j].values, self.clusters[i].centroid.postition.values)
            self.clusters[i].error /= len(self.clusters[i])
