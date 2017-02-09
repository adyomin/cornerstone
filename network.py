# Fully connected consecutive deep neural net

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    return np.multiply(x, (1 - x))


def pass_input(x):
    return x


def pass_input_prime(x):
    return 1


class Network:

    # TODO: add Network() docstring
    # TODO: add proper evaluation for epoch milestones

    """
    [TBD]
    """

    def __init__(self, features, targets, h_size, eta):

        """

        :param features:
        :param targets:
        :param h_size: tuple
        :param eta:
        """

        # features have to be of shape (n_records, n_features)
        # targets have to be of shape (n_records, n_targets)
        self._features = features
        self._targets = targets
        self._eta = eta
        self._forward_links = False
        self._backward_links = False

        self.n_features = features.shape[1]
        self.n_targets = targets.shape[1]
        self.shape = (features.shape[1], tuple(h_size), targets.shape[1])
        # depth takes into account only fully connected layers (i.e. ex input)
        self.depth = len(h_size) + 1

        self._layers = []
        # add input layer
        self._layers.append(Layer(self.n_features, activation=None,
                                  activation_prime=None))
        # add hidden layers
        for width in h_size:
            self._layers.append(Layer(width, activation=sigmoid,
                                      activation_prime=sigmoid_prime))
        del width

        # add output layer
        self._layers.append(Layer(self.n_targets, activation=pass_input,
                            activation_prime=pass_input_prime))

        # Link layers with each other, initialize weights
        for i in range(self.depth):
            # This part debugs fine: all next & prev. links are correct
            crt_layer = self._layers[i]
            nxt_layer = self._layers[i + 1]
            crt_layer._next = nxt_layer
            nxt_layer._previous = crt_layer

            # initializing weights w.r.t. actual structure
            # all layers but the input get weights assigned
            # This part debugs fine: weights seem to be correct (size, values)
            _scale = nxt_layer._previous._width**(-0.5)
            prev_width = nxt_layer._previous._width
            crt_width = nxt_layer._width
            _size = (prev_width, crt_width)
            nxt_layer._weights = np.random.normal(0, scale=_scale, size=_size)

            del crt_layer
            del nxt_layer
            del _scale
            del prev_width
            del crt_width
            del _size

    def train(self, batch_size, n_epochs):

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
                for j in range(1, self.depth + 1):
                    # Cant find other way to link it.  I guess all these
                    # links is a poor design choice in the first place, but at
                    # this point I am really concerned about the approaching
                    # deadline to invest more time in a better design :-(
                    current_layer = self._layers[j]
                    prev_layer = self._layers[j - 1]
                    current_layer._input = prev_layer._output
                    current_layer.forward()
                output_layer = self._layers[self.depth]

                # At this point all layers have produced their outputs, we need
                # to measure the error and start propagating it back to tune
                # the wights

                prediction = output_layer._output
                error = prediction - y
                cost = 0.5*(error**2)
                mse = ((2*cost).sum())/batch_size
                # d cost/d cost = 1
                # d cost/d error = (0.5*error**2)' = error
                # d error/d prediction = (prediction - error)' = 1
                output_layer._d_cost_d_output = error

                for k in range(self.depth, 0, -1):
                    current_layer = self._layers[k]
                    next_layer = self._layers[k - 1]
                    current_layer.backward(eta=self._eta)
                    next_layer._d_cost_output = current_layer._d_cost_d_input

                del cost
                del current_layer
                del error
                del input_layer
                del next_layer
                del output_layer
                del prediction
                del prev_layer
                del x
                del y

            if epoch%(n_epochs//10) == 0:
                print('epoch = {0}/{1}, MSE = {2:.5f}'.format(epoch, n_epochs,
                                                           mse))
        del data
        del max_len

    def evaluate(self):
        print('Not Yet Implemented')
        pass

    def save_weights(self):
        print('Not Yet Implemented')
        pass


class Layer:

    def __init__(self, width, activation=sigmoid,
                 activation_prime=sigmoid_prime):
        self._width = width
        self._activation = activation
        self._activation_prime = activation_prime
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
        self._d_output_d_arg = self._activation_prime(self._output)

        # (batch_size, self._width)
        # d cost/d arg = d cost/d output * d output/d arg
        if self._next is not None:
            # I don't know why I had to add this, does not make much sense to
            # me, as I saw the value in the debugger, but still there was a
            # None type error in np.multiply as if reference resolution was
            # bugged / not obvious?  I am really confused here.
            # Update: there was a potential mix up of for loop i-s.
            # TODO: check functionality without additional assignment
            self._d_cost_d_output = self._next._d_cost_d_input

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

        self._weights += (-1)*eta*self._d_cost_d_weights
