from DataStructure.Centroid import Centroid
from DataStructure.Data import Data
import numpy

class Cluster:

    def __init__(self, centroid, save_last_centroid):
        self.data = []
        self.centroid = centroid
        self.error = 0
        if save_last_centroid:
            neuron_position_copy = numpy.copy(centroid.position.values)
            tmp_data = Data(neuron_position_copy)
            self.previous_centroid = Centroid(tmp_data)
