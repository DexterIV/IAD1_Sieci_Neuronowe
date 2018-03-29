from DataStructure.Cluster import Cluster
from DataStructure.Neuron import Neuron
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
        self.data.clear()
        dataset = pd.read_csv(filename)

        sepal_length = dataset.iloc[:, 1].values
        sepal_width = dataset.iloc[:, 2].values
        petal_length = dataset.iloc[:, 3].values
        petal_width = dataset.iloc[:, 4].values

        for i in range(len(sepal_length)):
            values = [sepal_length[i], sepal_width[i], petal_length[i], petal_width[i]]
            self.data.append(Data(values))

    def initialize_centroids(self):
        for i in range(self.numberOfClusters):
            import random
            neurons_position = Data(self.data[random.randint(0, len(self.data) - 1)].values)
            neuron = Neuron(neurons_position)
            self.clusters.append(Cluster(neuron))

    def algorithm(self):
        i = 0
        for i in range(self.maxIterations):
            clear_clusters(self.clusters)
            self._assign_data_to_clusters()
            self._move_centroids()
            if i % 2 == 0:
                plot_all_clusters(self.clusters, i, 'KMeans algorithm')
            self._reassign_clusters_with_little_data()
            if self._second_stop_condition():
                break
        plot_all_clusters(self.clusters, i, 'KMeans algorithm')

    def _assign_data_to_clusters(self):
        for j in range(len(self.data)):
            closest_centroid_index = self._find_minimum_distance(self.data[j])
            self.clusters[closest_centroid_index].data.append(self.data[j])

    def _move_centroids(self):
        for i in range(len(self.clusters)):
            copy_values(self.clusters[i].neuron.position.values,
                        self.clusters[i].previous_neuron.position.values)
            new_centroid_position = []
            new_centroid_position.clear()
            new_centroid_position = [0] * len(self.data[0].values)
            for j in range(len(new_centroid_position)):
                for k in range(len(self.clusters[i].data)):
                    new_centroid_position[j] += self.clusters[i].data[k].values[j]
                new_centroid_position[j] /= len(self.clusters[i].data)
            copy_values(new_centroid_position, self.clusters[i].neuron.position.values)

    def _reassign_clusters_with_little_data(self):
        import random
        for i in range(self.numberOfClusters):
            if len(self.clusters[i].data) <= self.littleDataThreshold * len(self.data):
                self.clusters[i].neuron.position = self.data[random.randint(0, len(self.data) - 1)]

    def _find_minimum_distance(self, data_instance):
        index = 0
        minimum = distance(data_instance, self.clusters[index].neuron.position)
        for i in range(1, len(self.clusters)):
            dist = distance(self.clusters[i].neuron.position, data_instance)
            if dist < minimum:
                minimum = dist
                index = i

        return index

    def _second_stop_condition(self):
        result = True
        for i in range(self.numberOfClusters):
            if distance(self.clusters[i].neuron.position,
                        self.clusters[i].previous_neuron.position) > self.absoluteTolerance:
                result = False

        return result
