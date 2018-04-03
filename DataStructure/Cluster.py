from UniversalFunctions import *
from DataStructure.Centroid import Centroid
from DataStructure.Data import Data


class Cluster:

    def __init__(self, centroid, save_last_centroid):
        self.data = []
        self.centroid = centroid
        if save_last_centroid == True:
            neuron_position_copy = [0] * len(centroid.position.values)
            copy_values(centroid.position.values, neuron_position_copy)
            tmp_data = Data(neuron_position_copy)
            self.previous_centroid = Centroid(tmp_data)
