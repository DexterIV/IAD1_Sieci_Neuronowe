import math


class Data:
    def __init__(self, values):
        self.values = values

    def distance_from_neuron(self, neuron):
        distance = 0
        for i in range(len(self.values)):
            distance += (self.values[i] - neuron.position.values[i])**2

        distance = math.sqrt(distance)
        return distance
