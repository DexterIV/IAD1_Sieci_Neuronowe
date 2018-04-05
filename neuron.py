class Neuron:
    def __init__(self, position=[], data=[]):
        self._position = position
        self._data = data
        self.cluster = None

    def set_data(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def set_position(self, position):
        self._position = position

    def get_position(self):
        return self._position
