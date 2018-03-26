from DataStructure.Cluster import Cluster
from DataStructure.Neuron import Neuron
from DataStructure.Data import Data
import math


class NeuralGas:
    def __init__(self, number_of_clusters=3, max_iterations=128, learning_grade=0.1, learning_lambda=0.1,
                 absolute_tolerance=0.000001):
        self.clusters = []
        self.numberOfClusters = number_of_clusters
        self.maxIterations = max_iterations
        self.absoluteTolerance = absolute_tolerance
        self.data = []
        self.learning_grade = learning_grade
        self.learning_lambda = learning_lambda

    def next_neuron_position(self, neuron):
        return neuron + self.learning_grade*math.exp()

    def initialize_data(self, filename):
        self.data.clear ()
        dataset = pd.read_csv (filename)

        sepal_length = dataset.iloc[:, 1].values
        sepal_width = dataset.iloc[:, 2].values
        petal_length = dataset.iloc[:, 3].values
        petal_width = dataset.iloc[:, 4].values

        for i in range (len (sepal_length)):
            values = [sepal_length[i], sepal_width[i], petal_length[i], petal_width[i]]
            self.data.append(Data(values))

