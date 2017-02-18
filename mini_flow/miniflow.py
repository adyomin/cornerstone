import numpy as np


class Node(object):
    """
    Base class for nodes in the network.

    Parameters
    ----------

    inbound_nodes : list
        A list of nodes with edges into this node.
    """

    def __init__(self, inbound_nodes=[]):
        """
        Node's constructor (runs when the object is instantiated). Sets
        properties that all nodes need.
        """

        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # The eventual value of this node. Set by running the forward() method.
        self.value = None
        # Keys are the inputs to this node and their values are the partials
        # of this node with respect to that input.
        self.gradients = {}
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Every node that uses this class as a base class will need to define
        its own `forward` method.
        """

        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will need to define
        its own `backward` method.
        """
        raise NotImplementedError


class Input(Node):
    """
    A generic input into the network.
    """

    def __init__(self):
        # The base class constructor has to run to set all the properties here.
        #
        # The most important property on an Input is value. self.value is set
        #  during `topological_sort` later.
        Node.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input node has no inputs so the gradient (derivative) is zero.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


class Add(Node):
    def __init__(self, *inputs):
        # You could access `x` and `y` in forward with
        # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
        Node.__init__(self, inputs)

    def forward(self):
        """
        Set self.value to the sum of inbound inbound_nodes' values.
        """

        self.value = 0
        for node in self.inbound_nodes:
            self.value += node.value

    def backward(self):
        # d Add/d xi = (x1 + x2 + ... + xi + ... xn)' = 1
        # TODO: Implements Add.backward?
        raise NotImplementedError


class Mul(Node):
    # Using arguments unpacking (* for lists & tuples, ** for dictionaries)
    # docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists

    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Set self.value to the product of inbound inbound_nodes' values.
        """

        self.value = 1
        for node in self.inbound_nodes:
            self.value *= node.value

    def backward(self):
        # d Mul/d xi = (x1*x2* ... *xi* ... *xn)' = x1*x2* ... *xn
        # TODO: Implements Mul.backward?
        raise NotImplementedError

class Linear(Node):
    """
    Represents a node that performs a linear transform.
    """

    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
        # NOTE: The weights and bias properties here are not numbers,
        # but rather references to other nodes. The weight and bias values
        # are stored within the respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.
        """

        # assumed shape = (batch_size, input_width)
        inputs = self.inbound_nodes[0].value
        # assumed shape = (input_width, output_width)
        weights = self.inbound_nodes[1].value
        # assumed shape = (1, output_width)
        bias = self.inbound_nodes[2].value
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """

        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] \
                += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] \
                += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] \
                += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """

    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it will be used later
        with `backward` as well.

        x : A numpy array-like object.

        Return the result of the sigmoid function.
        """
        return 1/(1 + np.exp(-x))

    def forward(self):
        """
        Set the value of this node to the result of the sigmoid function,
        `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors if you test without
        # changing this method.
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # d cost/d x = d cost/d sig * d sig/d x
            # d cost/d sig.shape -> (batch_size, output_width)
            # d sig/d x.shape -> (batch_size, output_width)
            d_sig_d_x = self.value*(1 - self.value)
            self.gradients[self.inbound_nodes[0]] += np.multiply(grad_cost,
                                                                 d_sig_d_x)

class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.  Should be used as the last
        node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """

        # Don't like a typecasting here.
        y = np.array(self.inbound_nodes[0].value, ndmin=2).T
        a = self.inbound_nodes[1].value

        shape_error = 'y.shape is not equal to a.shape'
        assert y.shape == a.shape, shape_error

        # outputs vector or squared errors
        self.m = y.shape[0]
        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final node of the network so outbound nodes are not a
        concern.
        """
        # d cost/d y = c cost/d error * d error/d y
        # (batch_size, output_width)
        self.gradients[self.inbound_nodes[0]] = (2/self.m)*self.diff
        # d cost/d a = c cost/d error * d error/d a
        # (batch_size, output_width)
        self.gradients[self.inbound_nodes[1]] = (-2/self.m)*self.diff



def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm -
    algorithm's description.  Algorithm has running time linear in the number
    of nodes plus the number of edges, asymptotically, O(|V|+|E|).

    Parameters
    ----------

    feed_dict : dict
        A dictionary where the key is a `Input` node and the value is the
        respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Parameters
    ----------

    graph : list
        A topologically sorted list of nodes.
    """

    # Forward pass
    for node in graph:
        node.forward()

    # Backward pass
    for node in graph[::-1]:
        node.backward()
