# Fully connected consecutive deep neural net

import numpy as np


class Network:

    # TODO: add Network() docstring(s)

    def __init__(self, size, h_activation='sigmoid',
                 o_activation='pass_input', c_function='quadratic'):
        """ Network class constructor method.

        Parameters
        ----------
        size : tuple
        size[0] - n_features, input layers width.  size[-1] - n_targets,
        width of the output layer.

        h_activation : string
        Choice of activation function for all hidden layers.  Current options include:
         - 'sigmoid' - returns 1/(1 + exp(-x))
         - 'pass_input' - returns x

        o_activation : string
        Choice of output layer activation function.  Current options include:
         - 'sigmoid' - returns 1/(1 + exp(-x))
         - 'pass_input' - returns x

        c_function : string
        Choice of cost/loss/objective function prime for the network.
        Current options include:
         - 'quadratic' - returns error.  (0.5*(x**2))' = x.
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
        self._activation_list = [[] for layer in range(self.depth)]
        for i in range (self.depth - 1):
            self._activation_list[i] = h_activation
        self._activation_list[-1] = o_activation
        self._outputs = [[] for layer in range(self.depth)]
        self._d_cost_d_outputs = [[] for layer in range(self.depth)]
        self._tmp_value = None

    def _activation(self, x, function):
        if function == 'sigmoid':
            return 1/(1 + np.exp(-x))
        elif function == 'pass_input':
            return x

    def _activation_prime(self, x, function):
        if function == 'sigmoid':
            # x should be equal to the sigmoid output
            return np.multiply(x, (1 -x))
        elif function == 'pass_input':
            return 1

    def _cost_prime(self, x, function):
        # 'quadratic' = 0.5*(x**2)
        if function == 'quadratic':
            return x

    def _MSE(self, prediction, label):
        return np.mean((prediction - label) ** 2)

    def _forward(self, x):
        """ Calculates network prediction of y for x.

        Parameters
        ----------
        x : numpy.array
            Input should be of size (batch_size, n_features).  Typical
            features are real numbers with mean = 0, scaled to 1 standard
            deviation.  Use of non-numeric numbers is not supposed
            to produce any meaningful output.  Use of non-standardized
            features would most likely produce poor results.
        """

        exception_text_1 = 'n_features (x.shape[1]) is not equal to input ' \
                           'width (self.shape[0])'
        assert self.shape[0] == x.shape[1], exception_text_1

        layer_input = x
        # store input for backward pass
        self._tmp_value = x
        for i in range (0, self.depth):
            layer_weights = self._weights[i]
            a_function = self._activation_list[i]
            # (batch_size, input_width) @ (input_width, output_width) -> (
            # batch_size, output_width)
            layer_arg = np.dot(layer_input, layer_weights)
            self._outputs[i] = self._activation(layer_arg, function=a_function)
            layer_input = self._outputs[i]

    def _backward(self, y, eta = 0.01):
        """ Updates weights based on d cost/d output for each layer.

        Parameters
        ----------
        y : numpy.array
            y should be of size (batch_size, n_targets).

        eta : float
            Learning rate.  In some not so distant future would be nice to
            change it for something more advanced.  Adam?
        """

        exception_text_1 = 'n_targets (x.shape[1]) is not equal to output ' \
                           'width (self.shape[-1])'
        assert self.shape[-1] == y.shape[1], exception_text_1

        # (batch_size, output_width)
        # for the purpose of d_cost_d_output calculation it is important to
        # be consistent with the error form
        error = y - self._outputs[-1]

        # seed self._d_cost_d_outputs[-1]
        # (batch_size, output_width)
        # d cost/d error = (1/2*(error**2))' = error
        d_cost_d_error = self._cost_prime(error, function=self._cost_function)
        # d error/ d output = (y - output)' = -1
        d_error_d_output = -1
        d_cost_d_output = d_cost_d_error*d_error_d_output
        self._d_cost_d_outputs[-1] = d_cost_d_output

        # get the loop going
        for i in range(self.depth - 1, 0, -1):
            # (batch_size, output_width)
            output = self._outputs[i]
            d_cost_d_output = self._d_cost_d_outputs[i]
            a_function = self._activation_list[i]

            # (batch_size, output_width)
            # (!) Making calculations efficient for sigmoid I have to hard code
            # the optimization for now, making this part of the derivation
            # erroneous for all other activation functions (expect pass_input).
            # TODO: save local gradients during forward pass?
            d_output_d_arg = self._activation_prime(output, function=a_function)
            d_cost_d_arg = np.multiply(d_cost_d_output, d_output_d_arg)

            # (inout_width, output_width)
            # d arg/d input = (input * weights)' = weights
            d_arg_d_input = self._weights[i]

            # has to be (batch_size, input_width)
            # d cost/d input = d cost/d arg * d cost/d input
            d_cost_d_input = np.matmul(d_cost_d_arg, d_arg_d_input.T)
            # (!) this makes separate calculation for layer_0 necessary
            self._d_cost_d_outputs[i - 1] = d_cost_d_input

            # (batch_size, input_width)
            # d arg/d weights = (input * weights)' = input
            # (!) this makes separate calculation for layer_0 necessary
            d_arg_d_weights = self._outputs[i - 1]

            # (batch_size, input_width).T @ (batch_size, output_width) ->
            # (input_width, output_width)
            # d cost/d wights = d cost/d arg * d arg/d weights
            d_cost_d_weights = np.matmul(d_arg_d_weights.T, d_cost_d_arg)

            self._weights[i] += -eta*d_cost_d_weights

        # input layer weights update
        output = self._outputs[0]
        d_cost_d_output = self._d_cost_d_outputs[0]
        a_function = self._activation_list[0]
        d_output_d_arg = self._activation_prime(output, function=a_function)
        d_cost_d_arg = np.multiply(d_cost_d_output, d_output_d_arg)
        d_arg_d_weights = self._tmp_value
        d_cost_d_weights = np.matmul(d_arg_d_weights.T, d_cost_d_arg)
        self._weights[0] += -eta * d_cost_d_weights

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
