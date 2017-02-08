import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return x*(1 - x)

class Network():

    """
    TBD:
     - Complete doc string(s)
     - Add 'boilerplate' code (size & type assertions, exceptions, etc)
    """

    def __init__(self, features, targets, h_size, eta, activation=sigmoid,
                 activation_prime=sigmoid_prime):

        """

        :param features:
        :param targets:
        :param h_size: tuple
        :param activation:
        :param activation_prime:
        :param eta:
        """

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

        self.n_features = features.shape[1]
        self.n_targets = targets.shape[1]
        self.shape = (features.shape[1], tuple(h_size), targets.shape[1])
        # depth takes into account only fully connected layers (i.e. ex input)
        self.depth = len(h_size) + 1

        self._layers = []
        # add input layer
        self._layers.append(Layer(self.n_features))
        # add hidden layers
        for width in h_size:
            self._layers.append(Layer(width))
        # add output layer
        self._layers.append(Layer(self.n_targets))

        # link layers with each other
        for i in range(self.depth):
            crt_layer = self._layers[i]
            nxt_layer = self._layers[i + 1]
            crt_layer._next = nxt_layer
            nxt_layer._previous = crt_layer


    def train(self, batch_size, n_epochs, shuffle=True):

        """

        :param batch_size:
        :param n_epochs:
        :param shuffle:
        :return:
        """

        data = np.hstack((self._features, self._targets))
        max_len = self._features.shape[0]

        for epoch in range(n_epochs):
            np.random.shuffle(data)
            for i in range(0, max_len, batch_size):
                x = data[i:min(i+batch_size, max_len), :-self.n_targets]
                y = data[i:min(i+batch_size, max_len), -self.n_targets:]

            if epoch%(n_epochs//10) == 0:
                print('epoch = {0} / {1}, error = NEI'.format(epoch, n_epochs))

    def evaluate(self):
        pass

    def save_weights(self):
        pass

class Layer(Network):

    # TODO Layer.__init__ - adjust for no links instantiation

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