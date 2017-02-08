import numpy as np


class Network():

    """
    TBD:
     - Complete doc string(s)
     - Add 'boilerplate' code (size & type assertions, exceptions, etc)
    """

    def __init__(self, features, targets, h_size, activation,
                 activation_prime):

        # To-Do-List:
        # - add proper choice of activation function & activation_prime
        # - layer objects creation

        # features have to be of shape (n_records, n_features)
        # targets have to be of shape (n_records, n_targets)
        self._activation = activation
        self._activation_prime = activation_prime
        self.shape = (features.shape[1], tuple(h_size), targets.shape[1])
        self._features = features
        self._targets = targets

    def train(self):
        pass

    def evaluate(self):
        pass


class Layer(Network):

    def __init__(self, previous, next, width):
        self._previous = previous
        self._next = next
        self._width = width
        self._weights = np.random.normal(0,
                                         scale=previous._width**(-0.5),
                                         size= (previous._width, self._width)
                                         )

    def forward(self, batch):
        # batch has to be of size (batch_size, previous._width)
        pass

    def backward(self):
        pass