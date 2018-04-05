from DataStructure.Cluster import Cluster
from DataStructure.Neuron import Neuron
from DataStructure.Data import Data
from UniversalFunctions import *
import pandas as pd
import math


class Kohonen:
    def __init__(self, number_of_neurons=6, max_iterations=500, learning_grade=0.5, _lambda=20,
                 neighbourhood_radius=1.5, absolute_tolerance=0.0001):
        self.numberOfNeurons = number_of_neurons
        self.maxIterations = max_iterations
        self.data = []
        self.neurons = []
        self.initial_learning_grade = learning_grade
        self.learning_grade = learning_grade
        self.initial_neighbourhood_radius = neighbourhood_radius
        self.neighbourhood_radius = neighbourhood_radius
        self.wage_lambda = _lambda
        self.absolute_tolerance = absolute_tolerance
        self.saved_neurons = []
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

    def initialize_neurons(self):
        import random
        for i in range(self.numberOfNeurons):
            self.neurons.append(Neuron(i, self.data[random.randint(0, len(self.data) - 1)].values))

    def _weight_function(self, _distance):
        return math.exp(-(_distance ** 2) / (2 * (self.neighbourhood_radius ** 2)))

    def _neighbourhood_decay(self, iteration):
        self.neighbourhood_radius = self.initial_neighbourhood_radius * math.exp(- iteration / self.wage_lambda)

    def _learning_grade_decay(self, iteration):
        self.learning_grade = self.initial_learning_grade * math.exp(- iteration / self.wage_lambda)

    def algorithm(self):
        import random
        i = 0
        for i in range(self.maxIterations):
            self._save_neurons()
            sample = self.data[random.randint(0, len(self.data) - 1)]
            best_matching_unit = self.neurons[self._find_minimum_distance(sample)]
            for j in range(len(self.neurons)):
                self._calculate_new_weights(self.neurons[j], best_matching_unit, sample)
            self._learning_grade_decay(i)
            self._neighbourhood_decay(i)
            if self._second_stop_condition(i):
                break
        plot_all_clusters(self._define_clusters_for_plotting(), i, 'Kohonen\'s self-organizing map',
                          len(self.data[0].values), self.saved_neurons, self.dataLabels)

    def _calculate_new_weights(self, neuron, bmu, sample):
        if neuron == bmu:
            for i in range(len(neuron.weights)):
                bmu.weights[i] += self.learning_grade * (sample.values[i] - bmu.weights[i])
        else:
            dist = abs(neuron.position - bmu.position)
            if dist < self.neighbourhood_radius:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += self.learning_grade * (sample.values[i] - neuron.weights[i]) * \
                                         self._weight_function(dist)

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

    def _save_neurons(self):
        saved = []
        for i in range(len(self.neurons)):
            copy = [0] * len(self.neurons[0].weights)
            copy_values(self.neurons[i].weights, copy)
            saved.append(Neuron(i, copy))
        self.saved_neurons.append(saved)

    def _second_stop_condition(self, iteration):
        result = True
        for i in range(len(self.neurons)):
            if distance(self.neurons[i].weights,
                        self.saved_neurons[iteration][i].weights) > self.absolute_tolerance:
                result = False
        return result
