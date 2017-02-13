# Fully connected consecutive deep neural net

import numpy as np


class Network:

    # TODO: add Network() docstring(s)

    def __init__(self, size, h_activation='sigmoid',
                 o_activation='pass_input', c_function='quadratic'):
        """

        :param size: tuple; size[0] - n_features, input layers width.  size[-1]
         - n_targets, width of the output layer.
        """

        self.shape = size
        # depth takes into account only fully connected layers (i.e. ex input)
        self.depth = len(size) - 1

        self._cost_function = c_function
        self._weights = []
        for i in range(0, self.depth, 1):
            input_width = size[i]
            output_width = size[i + 1]
            scale = input_width**(-0.5)
            wm_size = (input_width, output_width)
            self._weights.append(np.random.normal(0, scale=scale, size=wm_size))
        # add activation functions list (string elements)
        self._activation_list = []
        for i in range (self.depth - 1):
            self._activation_list.append(h_activation)
        self._activation_list.append(o_activation)
        self._outputs = []
        self._d_cost_d_outputs = []

    def _activation(self, x, function):
        if function == 'sigmoid':
            return 1/(1 + np.exp(-x))
        elif function == 'pass_input':
            return x

    def _activation_prime(self, x, function):
        if function == 'sigmoid':
            # TODO: make this obvious (docstring?)
            # x should be equal to the sigmoid output
            return np.multiply(x, (1 -x))
        elif function == 'pass_input':
            return 1

    def _cost_prime(self, x, function):
        # 'quadratic' = 0.5*(x**2)
        if function == 'quadratic':
            return x

    def MSE(self, prediction, label):
        return np.mean((prediction - label) ** 2)

    def forward(self, x):
        # input should be of size (batch_size, n_features)
        exception_text_1 = 'n_features (x.shape[1]) is not equal to input ' \
                           'width (self.shape[0])'
        assert self.shape[0] == x.shape[1], exception_text_1

        layer_input = x
        for i in range (0, self.depth):
            layer_weights = self._weights[i]
            a_function = self._activation_list[i]
            # (batch_size, input_width) @ (input_width, output_width) -> (
            # batch_size, output_width)
            layer_arg = np.dot(layer_input, layer_weights)
            self._outputs[i] = self._activation(layer_arg, function=a_function)
            layer_input = self._outputs[i]

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
                # d cost/d cost = 1
                # d cost/d error = (0.5*error**2)' = error
                # d error/d prediction = (prediction - target)' = 1
                output_layer._d_cost_d_output = error

                for k in range(self.depth, 0, -1):
                    current_layer = self._layers[k]
                    next_layer = self._layers[k - 1]
                    current_layer.backward(eta=self._eta)
                    next_layer._d_cost_output = current_layer._d_cost_d_input

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
                x = self._features
                y = self._targets
                mse = self.evaluate(x, y)
                print('epoch = {0}/{1}, MSE = {2:.5f}'.format(epoch, n_epochs,
                                                           mse))
        del data
        del max_len

    def evaluate(self, x, y):
        input_layer = self._layers[0]
        input_layer._output = x
        for j in range(1, self.depth + 1):
            current_layer = self._layers[j]
            prev_layer = self._layers[j - 1]
            current_layer._input = prev_layer._output
            current_layer.forward()
        output_layer = self._layers[self.depth]
        predictions = output_layer._output
        return MSE(predictions, y)

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
