from DataStructure.Cluster import Cluster
from DataStructure.Neuron import Neuron
from DataStructure.Data import Data
from UniversalFunctions import *
import pandas as pd
import math


class Kohonen:
    def __init__(self, number_of_neurons=6, max_iterations=128, learning_grade=0.5, _lambda=20,
                 neighbourhood_radius=1.5):
        self.numberOfNeurons = number_of_neurons
        self.maxIterations = max_iterations
        self.data = []
        self.neurons = []
        self.initial_learning_grade = learning_grade
        self.learning_grade = learning_grade
        self.initial_neighbourhood_radius = neighbourhood_radius
        self.neighbourhood_radius = neighbourhood_radius
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
        import random
        for i in range(self.numberOfNeurons):
            self.neurons.append(Neuron(i, self.data[random.randint(0, len(self.data) - 1)]))

    def wage_function(self, _distance):
        return math.exp(-(_distance ** 2) / (2 * (self.neighbourhood_radius ** 2)))

    def neighbourhood_decay(self, iteration):
        self.neighbourhood_radius = self.initial_neighbourhood_radius * math.exp(- iteration / self.wage_lambda)

    def learning_grade_decay(self, iteration):
        self.learning_grade = self.initial_learning_grade * math.exp(- iteration / self.wage_lambda)

    def algorithm(self):
        import random
        i = 0
        for i in range(self.maxIterations):
            sample = self.data[random.randint(0, len(self.data) - 1)]
            BMU = self.neurons[self._find_minimum_distance(sample)]
            for j in range(len(self.neurons)):
                self._calculate_new_weights(self.neurons[j], BMU, sample)
            self.learning_grade_decay(i)
            self.neighbourhood_decay(i)
            if i % 2 == 0:
                self._plot_result(i)
        self._plot_result(i)

    def _calculate_new_weights(self, neuron, BMU, sample):
        if neuron == BMU:
            for i in range(len(neuron.weights)):
                BMU.weights[i] += self.learning_grade * (sample.values[i] - BMU.weights[i])
        else:
            dist = abs(neuron.position - BMU.position)
            if dist < self.neighbourhood_radius:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += self.learning_grade * (sample.values[i] - neuron.weights[i]) * \
                                        self.wage_function(dist)

    def _find_minimum_distance(self, data_instance):
        index = 0
        minimum = distance(data_instance.values, self.neurons[0].weights)
        for i in range(1, len(self.neurons)):
            dist = distance(self.neurons[i].weights, data_instance.values)
            if dist < minimum:
                minimum = dist
                index = i

        return index

    def _define_clusters_for_plotting(self):
        clusters = []
        for i in range(len(self.neurons)):
            clusters.append(Cluster(self.neurons[i], False))
        for j in range(len(self.data)):
            closest_neuron_index = self._find_minimum_distance(self.data[j])
            clusters[closest_neuron_index].data.append(self.data[j])
        return clusters

    def _plot_result(self, iteration):
        clusters = self._define_clusters_for_plotting()
        pyplot.figure('Kohonen\'s self-organizing map')
        pyplot.subplot(211)
        colors = ['b', 'g', 'c', 'xkcd:orchid', 'y', 'k', 'tab:olive', 'tab:pink', 'xkcd:coral', 'xkcd:indigo']
        for j in range (len (clusters)):
            self._plot_cluster (clusters[j], colors[j], 0, 1)
        pyplot.title('iteration no. ' + str (iteration + 1))
        pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
        pyplot.subplot(212)
        for j in range (len(clusters)):
            self._plot_cluster(clusters[j], colors[j], 2, 3)
        pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
        pyplot.show()

    def _plot_cluster(self, cluster, color, value_x_index, value_y_index):
        x = []
        y = []
        for j in range(len(cluster.data)):
            x.append(cluster.data[j].values[value_x_index])
            y.append(cluster.data[j].values[value_y_index])
        pyplot.plot(x, y, color=color, marker='x', linestyle='None')
        pyplot.plot(cluster.centroid.weights[value_x_index],
                    cluster.centroid.weights[value_y_index],
                    'ro')
