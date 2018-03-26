from UniversalFunctions import *
from DataStructure.Neuron import Neuron
from DataStructure.Data import Data


class Cluster:

    def __init__(self, neuron):
        self.data = []
        self.neuron = neuron
        neuron_position_copy = [0] * len(neuron.position.values)
        copy_values(neuron.position.values, neuron_position_copy)
        tmp_data = Data(neuron_position_copy)
        self.previous_neuron = Neuron(tmp_data)
