import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return x*(1 - x)

class Network():

    # TODO: add Network() docstring
    """
    [TBD]
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
        # TODO: (!) change output layer activation function to y=x
        # TODO: add proper choice of activation function & activation_prime

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

        # link layers with each other (init. weights, link outputs & d_costs)
        for i in range(self.depth):
            crt_layer = self._layers[i]
            nxt_layer = self._layers[i + 1]
            crt_layer._next = nxt_layer
            nxt_layer._previous = crt_layer

            # initializing weights w.r.t. actual structure
            # all layers but the input get weights assigned
            _scale = nxt_layer._previous._width**(-0.5)
            prev_width = nxt_layer._previous._width
            crt_width = nxt_layer._width
            _size = (prev_width, crt_width)
            nxt_layer._weights = np.random.normal(0, scale=_scale, size=_size)

            # setting up layers' output forward flow
            # all layers but the first get an input from each other
            nxt_layer._input = crt_layer._output

            # setting up cost function derivative backward flow
            # all layers but the last get a derivative from each other
            crt_layer._d_cost_d_output = nxt_layer._d_cost_d_input



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
                input_layer = self._layers[0]
                input_layer._output = x
                for i in range(1, self.depth + 1):
                    next_layer = self._layers[i]
                    next_layer.forward()
                output_layer = self._layers[self.depth]

                # At this point all layers have produced their outputs, we need
                # to measure the error and start propagating it back to tune
                # the wights

                prediction = output_layer._output
                error = prediction - y
                cost = 0.5*(error**2)
                mse = ((cost*2).sum())/batch_size
                # d cost/d cost = 1
                # d cost/d error = (0.5*error**2)' = error
                # d error/d prediction = (prediction - error)' = 1
                output_layer._d_cost_d_output = error

                for i in range(self.depth, 0, -1):
                    current_layer = self._layers[i]
                    current_layer.backward(eta=self._eta)

            if epoch%(n_epochs//10) == 0:
                print('epoch = {0}/{1}, MSE = {2}'.format(epoch, n_epochs, mse))

    def evaluate(self):
        print('Not Yet Implemented')
        pass

    def save_weights(self):
        print('Not Yet Implemented')
        pass

class Layer(Network):

    def __init__(self, width):
        self._width = width
        self._previous = None
        self._next = None
        self._input = None
        self._weights = None
        self._arg = None
        self._output = None
        self._d_cost_d_output = None
        self._d_output_d_arg = None
        self._d_cost_d_arg = None
        self._d_arg_d_input = None
        self._d_cost_d_input = None
        self._d_arg_d_weights = None
        self._d_cost_d_weights = None

    def forward(self):

        """
        Updates current layer output based on its wights and previous
        layer's output.

        """

        # (batch_size, self._width)
        self._arg = np.dot(self._input, self._weights)
        self._output = self._activation(self._arg)

    def backward(self, eta=0.001):

        """
        Updates layer's weights based on d cost/d output and passes d cost/d
        input down the line.

        Parameters
        ----------

        eta : float
            Learning rate.  In some not so distant future would be nice to
            change it for something more advanced.  Adam?

        """

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