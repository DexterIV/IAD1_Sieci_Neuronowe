from DataStructure.Cluster import Cluster
from DataStructure.Neuron import Neuron
from DataStructure.Data import Data


class Kohonen:
    def __init__(self, number_of_clusters=3, max_iterations=128):
        self.numberOfClusters = number_of_clusters
        self.maxIterations = max_iterations
