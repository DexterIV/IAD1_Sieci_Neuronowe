from Cluster import Cluster
from Neuron import Neuron
from Data import Data

import pandas as pd
import copy


class KMeans:
    def __init__(self, number_of_clusters=3, max_iterations=3, relative_tolerance=0.000001):
        self.clusters = []
        self.numberOfClusters = number_of_clusters
        self.maxIterations = max_iterations
        self.absoluteTolerance = relative_tolerance
        self.data = []

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
            neurons_position = copy.copy(self.data[i])
            neuron = Neuron(neurons_position)
            self.clusters.append(Cluster(neuron))

    def algorithm(self):
        for i in range(self.maxIterations):
            for j in range(len(self.data)):
                closest_centroid_index = self._find_minimum_distance(self.data[j])
                self.clusters[closest_centroid_index].data.append(self.data[j])
            for j in range(self.numberOfClusters):
                movement_vector = []
                movement_vector.clear()
                for k in range(len(self.clusters[j].data)):
                    movement_vector = [0] * len(self.data[0].values)
                    for l in range(len(self.data[0].values)):
                        movement_vector[l] += self.clusters[j].data[k].distance_from_neuron(self.clusters[j].neuron)
                for l in range(len(movement_vector)):
                    movement_vector[l] /= len(self.clusters[j].data)
                    self.clusters[j].neuron.position.values[l] += movement_vector[l]

    def _find_minimum_distance(self, data_instance):
        index = 0
        minimum = data_instance.distance_from_neuron(self.clusters[index].neuron)
        for i in range(1, len(self.clusters)):
            distance = data_instance.distance_from_neuron(self.clusters[i].neuron)
            if distance < minimum:
                minimum = distance
                index = i

        return index
