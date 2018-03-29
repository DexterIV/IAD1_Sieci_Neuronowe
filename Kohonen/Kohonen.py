from DataStructure.Cluster import Cluster
from DataStructure.Neuron import Neuron
from DataStructure.Data import Data
from UniversalFunctions import *
import pandas as pd
import math


class Kohonen:
    def __init__(self, number_of_clusters=3, max_iterations=128, learning_grade=0.05, _lambda=0.5, absolute_tolerance=0.000001):
        self.numberOfClusters = number_of_clusters
        self.maxIterations = max_iterations
        self.data = []
        self.neurons = []
        self.absolute_tolerance = absolute_tolerance
        self.learning_grade = learning_grade
        self.wage_lambda = _lambda

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

    def initialize_neurons(self):
        for i in range(self.numberOfClusters):
            import random
            neurons_position = Data(self.data[random.randint(0, len(self.data) - 1)].values)
            self.neurons.append(Neuron(neurons_position))

    def wage_function(self, _distance):
        return math.exp(-(_distance ** 2) / (2 * self.wage_lambda))

    def algorithm(self):
        i = 0
        for i in range(self.maxIterations):
            oldNeurons = []
            for j in range(len(self.neurons)):
                tmp_values = [0] * len(self.neurons[j].position.values)
                tmp_position = Data(tmp_values)
                copy_values(self.neurons[j].position.values, tmp_position.values)
                oldNeurons.append(Neuron(tmp_position))
                for k in range(len(self.data)):
                    for l in range(len(self.neurons[j].position.values)):
                        movement_vector = self.calculate_movement_vector(self.neurons[j], self.data[k])
                        self.neurons[j].position.values[l] += movement_vector[l]
            if i % 2 == 0:
                clusters = self.define_clusters_for_plotting()
                plot_all_clusters(clusters, i, 'Kohonen\'s net algorithm')
            if self._second_stop_condition(oldNeurons):
                break
            oldNeurons.clear()
        clusters = self.define_clusters_for_plotting()
        plot_all_clusters(clusters, i, 'Kohonen\'s net algorithm')

    def calculate_movement_vector(self, neuron, data_instance):
        vector = []
        for i in range(len(neuron.position.values)):
            tmp = data_instance.values[i] - neuron.position.values[i]
            tmp = self.learning_grade * self.wage_function(tmp)
            vector.append(tmp)
        return vector

    def define_clusters_for_plotting(self):
        clusters = []
        for i in range(len(self.neurons)):
            clusters.append(Cluster(self.neurons[i]))
        for j in range(len(self.data)):
            closest_centroid_index = self._find_minimum_distance(self.data[j])
            clusters[closest_centroid_index].data.append(self.data[j])
        return clusters

    def _find_minimum_distance(self, data_instance):
        index = 0
        minimum = distance(data_instance, self.neurons[index].position)
        for i in range(1, len(self.neurons)):
            dist = distance(self.neurons[i].position, data_instance)
            if dist < minimum:
                minimum = dist
                index = i

        return index

    def _second_stop_condition(self, old_neurons):
        result = True
        for i in range(self.numberOfClusters):
            if distance(self.neurons[i].position,
                        old_neurons[i].position) > self.absolute_tolerance:
                result = False

        return result
