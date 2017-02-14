# Fully connected consecutive deep neural net

import numpy as np


class Network:

    def __init__(self, size, h_activation='sigmoid',
                 o_activation='pass_input', c_function='quadratic',
                 weights=None):
        """ Network class constructor method.

        Parameters
        ----------
        size : tuple
            size[0] - n_features, input layers width.  size[-1] - n_targets,
            width of the output layer.

        h_activation : string
            Choice of activation function for all hidden layers.  Current
            options include:
             - 'sigmoid' - returns 1/(1 + numpy.exp(-x))
             - 'pass_input' - returns x

        o_activation : string
            Choice of output layer activation function.  Current options
            include:
             - 'sigmoid' - returns 1/(1 + numpy.exp(-x))
             - 'pass_input' - returns x

        c_function : string
            Choice of cost/loss/objective function prime for the network.
            Current options include:
             - 'quadratic' - returns error, (0.5*(error**2))' = error

        weights : numpy.array - optional
            Create a network instance using previously saved weights.
            Dimensions control is on the user atm.
        """

        self.shape = size
        # depth takes into account only fully connected layers (i.e. ex input)
        self.depth = len(size) - 1
        self._cost_function = c_function
        if weights is None:
            self._weights = []
            for i in range(0, self.depth, 1):
                input_width = size[i]
                output_width = size[i + 1]
                scale = input_width**(-0.5)
                wm_size = (input_width, output_width)
                self._weights.append(np.random.normal(0, scale=scale,
                                                      size=wm_size))
        else:
            self._weights = weights
        # add activation functions list (string elements)
        self._activation_list = [[] for layer in range(self.depth)]
        for i in range(self.depth - 1):
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
            return np.multiply(x, (1 - x))
        elif function == 'pass_input':
            return 1

    def _cost_prime(self, x, function):
        # 'quadratic' = 0.5*(x**2)
        if function == 'quadratic':
            return x

    def _forward(self, x):
        """ Calculates network prediction of y for x.

        Parameters
        ----------
        x : numpy.array
            Train examples matrix of size (batch_size, n_features).  Matrix
            elements are expected to be real numbers with mean = 0.0,
            scaled to 1.0 standard deviation.
        """

        exception_text_x = 'n_features (x.shape[1]) is not equal to input ' \
                           'width (self.shape[0])'
        assert self.shape[0] == x.shape[1], exception_text_x

        layer_input = x
        # store input for backward pass
        self._tmp_value = x
        for i in range(0, self.depth):
            layer_weights = self._weights[i]
            a_function = self._activation_list[i]
            # (batch_size, input_width) @ (input_width, output_width) -> (
            # batch_size, output_width)
            layer_arg = np.dot(layer_input, layer_weights)
            self._outputs[i] = self._activation(layer_arg, function=a_function)
            layer_input = self._outputs[i]

    def _backward(self, y, batch_size, eta=0.01):
        """ Updates weights based on d cost/d output for each layer.

        Parameters
        ----------
        y : numpy.array
            y should be of size (batch_size, n_targets).

        eta : float
            Learning rate.  In some not so distant future would be nice to
            change it for something more advanced.  Adam?
        """

        exception_text_y = 'n_targets (y.shape[1]) is not equal to output ' \
                           'width (self.shape[-1])'
        assert self.shape[-1] == y.shape[1], exception_text_y

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
            self._weights[i] += -eta*d_cost_d_weights/batch_size

        # input layer weights update
        output = self._outputs[0]
        d_cost_d_output = self._d_cost_d_outputs[0]
        a_function = self._activation_list[0]
        d_output_d_arg = self._activation_prime(output, function=a_function)
        d_cost_d_arg = np.multiply(d_cost_d_output, d_output_d_arg)
        # d arg/d weights = (input * weights)' = input
        # using x saved @ train() step
        d_arg_d_weights = self._tmp_value
        d_cost_d_weights = np.matmul(d_arg_d_weights.T, d_cost_d_arg)
        self._weights[0] += -eta*d_cost_d_weights/batch_size

    def train(self, x_train, y_train, batch_size, eta, n_epochs, shuffle=True):
        """ Trains the instance with its current weights to predict y from x.

        Parameters
        ----------
        x_train : numpy.array
            Train examples matrix of size (n_records, n_features).  Matrix
            elements are expected to be real numbers with mean = 0.0, scaled
            to 1.0 standard deviation.

        y_train : numpy.array
            Train labels matrix of size (n_records, n_targets).

        batch_size : int
            Defines how many train examples will be taken for the next step of
            the weights update.  Would be great to add some intuition how to
            choose a batch size.  TBD I guess.

        n_epochs : int
            Number of epochs to train over the whole x.

        eta : float
            Weight update step multiple.  Constant only ATM.

        shuffle: bool
            Determines whether or not data is being shuffled each epoch.
        """

        exception_x = 'n_features (x_train.shape[1]) is not equal to input ' \
                      'width self.shape[0])'
        assert self.shape[0] == x_train.shape[1], exception_x
        exception_y = 'n_targets (y_train.shape[1]) is not equal to output ' \
                      'width (self.shape[-1])'
        assert self.shape[-1] == y_train.shape[1], exception_y
        exception_n_records = 'x_train.shape[0] is not equal to y_train.shape[0]'
        assert x_train.shape[0] == y_train.shape[0], exception_n_records
        exception_ndim = 'len(x_train.shape) is not equal to len(y_train.shape)'
        assert len(x_train.shape) == len(y_train.shape), exception_ndim

        data = np.hstack((x_train, y_train))
        n_records = x_train.shape[0]
        n_targets = y_train.shape[1]
        for epoch in range(n_epochs):
            if shuffle:
                np.random.shuffle(data)
            for i in range(0, n_records, batch_size):
                x = data[i:min(i + batch_size, n_records), :-n_targets]
                y = data[i:min(i + batch_size, n_records), -n_targets:]
                self._forward(x)
                self._backward(y, batch_size=batch_size, eta=eta)
            if epoch % (n_epochs//10) == 0:
                mse = self.evaluate(x_train, y_train)
                print('epoch = {0}/{1}, MSE = {2:.4f}'.format(epoch,
                                                              n_epochs, mse))
        mse = self.evaluate(x_train, y_train)
        print('epoch = {0}/{0}, MSE = {1:.4f}'.format(n_epochs, mse))

    def evaluate(self, x, y):
        """Returns Mean Squared Error of the network for y over x."""

        exception_x = 'n_features (x.shape[1]) is not equal to input width (' \
                      'self.shape[0])'
        assert self.shape[0] == x.shape[1], exception_x
        exception_y = 'n_targets (y.shape[1]) is not equal to output width (' \
                      'self.shape[-1])'
        assert self.shape[-1] == y.shape[1], exception_y
        exception_n_records = 'x.shape[0] is not equal to y.shape[0]'
        assert x.shape[0] == y.shape[0], exception_n_records
        exception_ndim = 'len(x.shape) is not equal to len(y.shape)'
        assert len(x.shape) == len(y.shape), exception_ndim

        self._forward(x)
        prediction = self._outputs[-1]
        return np.mean((prediction - y)**2)

    def predict(self, x):
        """Predicts labels for x using current weights."""

        exception_x = 'n_features (x.shape[1]) is not equal to input ' \
                      'width (self.shape[0])'
        assert self.shape[0] == x.shape[1], exception_x

        self._forward(x)
        pred_labels = self._outputs[-1]
        return pred_labels

    def get_weights(self):
        """Returns current network weights."""
        return self._weights

    def _save_weights(self):
        """Not yet implemented.  Public method eventually."""
        print('Not Yet Implemented')
        pass
