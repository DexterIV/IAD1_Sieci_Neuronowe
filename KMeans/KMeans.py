from DataStructure.Cluster import Cluster
from DataStructure.Neuron import Neuron
from DataStructure.Data import Data
from multiprocessing import Process

import pandas as pd
import copy
import matplotlib.pyplot as pyplot
import math


class KMeans:
    def __init__(self, number_of_clusters=3, max_iterations=256, absolute_tolerance=0.000001,
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
            neurons_position = copy.deepcopy(self.data[random.randint(0, len(self.data) - 1)])
            neuron = Neuron(neurons_position)
            self.clusters.append(Cluster(neuron))

    def algorithm(self):
        for i in range(self.maxIterations):
            self._assign_data_to_centroids()
            self._move_centroids()
            if (i + 1) % 64 == 0:
                KMeans._plot_all_clusters(self.clusters, i)
            self._reassign_clusters_with_little_data()
            KMeans._clear_clusters(self.clusters)
            if self._second_stop_condition():
                break

    def _assign_data_to_centroids(self):
        for j in range(len(self.data)):
            closest_centroid_index = self._find_minimum_distance (self.data[j])
            self.clusters[closest_centroid_index].data.append (self.data[j])

    def _move_centroids(self):
        for j in range(self.numberOfClusters):
            movement_vector = []
            movement_vector.clear ()
            for k in range(len(self.clusters[j].data)):
                movement_vector = [0] * len (self.data[0].values)
                for l in range(len(self.data[0].values)):
                    movement_vector[l] += KMeans.distance (self.clusters[j].neuron.position,
                                                           self.clusters[j].data[k])
            for l in range(len(movement_vector)):
                movement_vector[l] /= len (self.clusters[j].data)
                self.clusters[j].previousNeuron = copy.deepcopy(self.clusters[j].neuron)
                self.clusters[j].neuron.position.values[l] += movement_vector[l]

    @staticmethod
    def _clear_clusters(clusters):
        for i in range(len(clusters)):
            clusters[i].data.clear()

    def _reassign_clusters_with_little_data(self):
        import random
        for i in range(self.numberOfClusters):
            if len(self.clusters[i].data) <= self.littleDataThreshold * len(self.data):
                self.clusters[i].neuron.position = self.data[random.randint(0, len(self.data) - 1)]

    @staticmethod
    def distance(position1, position2):
        distance = 0

        if len(position1.values) != len(position2.values):
            return -1.0

        for i in range(len(position1.values)):
            distance += (position1.values[i] - position2.values[i]) ** 2

        distance = math.sqrt(distance)
        return distance

    def _find_minimum_distance(self, data_instance):
        index = 0
        minimum = self.distance(data_instance, self.clusters[index].neuron.position)
        for i in range(1, len(self.clusters)):
            distance = KMeans.distance(self.clusters[i].neuron.position, data_instance)
            if distance < minimum:
                minimum = distance
                index = i

        return index

    @staticmethod
    def _plot_cluster(cluster, color, value_x_index, value_y_index):
        x = []
        y = []
        for j in range(len(cluster.data)):
            x.append(cluster.data[j].values[value_x_index])
            y.append(cluster.data[j].values[value_y_index])
        pyplot.plot(x, y, color + 'x')
        pyplot.plot(cluster.neuron.position.values[value_x_index],
                  cluster.neuron.position.values[value_y_index],
                  'ro')

    @staticmethod
    def _plot_all_clusters(clusters, iteration):
        pyplot.figure('KMeans algorithm')
        pyplot.subplot(211)
        colors = ['b', 'g', 'm']
        for j in range(len(clusters)):
            KMeans._plot_cluster(clusters[j], colors[j], 0, 1)
        pyplot.title ('iteration no. ' + str (iteration + 1))
        pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
        pyplot.subplot(212)
        for j in range(len(clusters)):
            KMeans._plot_cluster(clusters[j], colors[j], 2, 3)
        pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
        pyplot.show()

    def _second_stop_condition(self):
        result = True
        for i in range(self.numberOfClusters):
            if KMeans.distance(self.clusters[i].neuron.position,
                               self.clusters[i].previousNeuron.position) > self.absoluteTolerance:
                result = False

        return result
