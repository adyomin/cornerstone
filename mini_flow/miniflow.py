import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # an Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

    # NOTE: Input node is the only node that may
    # receive its value as an argument to forward().
    #
    # All other node implementations should calculate their
    # values from the value of previous nodes, using
    # self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        if value is not None:
            self.value = value


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

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        # assumed shape = (batch_size, input_width)
        inputs = np.array(self.inbound_nodes[0].value, ndmin=2)
        # assumed shape = (output_width, input_width)
        # preferred shape = (input_width, output_width)
        weights = np.array(self.inbound_nodes[1].value, ndmin=2).T
        # assumed shape = (1, 1)
        bias = self.inbound_nodes[2].value
        self.value = np.dot(inputs, weights) + bias

"""
No need to change anything below here!
"""


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


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Parameters
    ----------

    output_node : type TBC
        A node in the graph, should be the output node (have no outgoing edges).
    sorted_nodes : type TBC
        A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
