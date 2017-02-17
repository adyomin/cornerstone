"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from mini_flow import miniflow as mf

x, y = mf.Input(), mf.Input()

f = mf.Add(x, y)

feed_dict = {x: 10, y: 5}

sorted_nodes = mf.topological_sort(feed_dict)
output = mf.forward_pass(f, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we
# could # also access the value for x with x.value (same goes for y).
print("{0} + {1} = {2} (according to miniflow)".format(feed_dict[x],
                                                       feed_dict[y], output))
