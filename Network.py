import numpy as np


class Network():

    """
    TBD:
     - Complete doc string(s)
     - Add 'boilerplate' code (size & type assertions, exceptions, etc)
    """

    def __init__(self, features, targets, h_size, activation,
                 activation_prime, eta):

        # To-Do-List:
        # - add proper choice of activation function & activation_prime
        # - layer objects creation

        # features have to be of shape (n_records, n_features)
        # targets have to be of shape (n_records, n_targets)
        self._activation = activation
        self._activation_prime = activation_prime
        self._features = features
        self._targets = targets
        self._eta = eta

        self.shape = (features.shape[1], tuple(h_size), targets.shape[1])

    def train(self):
        pass

    def evaluate(self):
        pass


class Layer(Network):

    def __init__(self, width, previous=None, next=None):
        self._previous = previous
        self._next = next
        self._width = width
        if previous != None:
            self._weights = np.random.normal(0,
                                             scale=previous._width**(-0.5),
                                             size=(previous._width, width)
                                            )
        if previous != None:
            self._input = previous._output
        else:
            self._input = None
        self._arg = None
        self._output = None
        if next != None:
            # (batch_size, self._width)
            self._d_cost_d_output = next._d_cost_d_input
        else:            # This looks bad :-(
            self._d_cost_d_output = None
        self._d_output_d_arg = None
        self._d_cost_d_arg = None
        self._d_arg_d_input = None
        self._d_cost_d_input = None
        self._d_arg_d_weights = None
        self._d_cost_d_weights = None

    def forward(self, batch=None):

        """
        Updates current layer output based on its wights and previous
        layer's output.

        Parameters
        ----------

        batch : numpy.array
            Has to be of size (batch_size, self.shape[0]).  At the
            moment it is only needed for input layer (self._previous ==
            None).  Has to be numpy.array or similar.

        Return
        ----------

        None.
        """

        if self._previous != None:
            # (batch_size, self._width)
            self._arg = np.dot(self._input, self._weights)
            self._output = self._activation(self._arg)
        else:
            self._output = batch


    def backward(self, eta=0.001, d_cost_d_output=None):

        """
        Updates layer's weights based on d cost/d output and passes d cost/d
        input down the line.

        Parameters
        ----------

        eta : float
            Learning rate.

        d_cost_d_output: numpy.array
            Has to be of size (batch_size, self.width).  At the
            moment it is only needed for the output layer (self._next ==
            None).  Has to be numpy.array or similar.

        Return
        ----------

        None.
        """

        if self._next == None:
            # (batch_size, self._width)
            self._d_cost_d_output = d_cost_d_output

        # (batch_size, self._width)
        self._d_output_d_arg = self._activation_prime(self._arg)

        # (batch_size, self._width)
        # d cost/d arg = d cost/d output * d output/d arg
        self._d_cost_d_arg = np.multiply(self._d_cost_d_output,
                                         self._d_output_d_arg)

        # (previous._width, self._width)
        self._d_arg_d_input = self._weights

        # has to be (batch_size, previous._width)
        # d cost/d input = d cost/d arg * d cost/d input
        self._d_cost_d_input = np.matmul(self._d_cost_d_arg,
                                         self._d_arg_d_input.T)
        # (batch_size, previous._width)
        self._d_arg_d_weights = self._input

        # has to be (previous._width, self._width)
        # d cost/d wights = d cost/d arg * d arg/d weights
        self._d_cost_d_weights = np.matmul(self._d_arg_d_weights.T,
                                           self._d_cost_d_arg)

        self._weights += -eta*self._d_cost_d_weights