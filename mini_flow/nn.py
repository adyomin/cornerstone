from mini_flow import miniflow as mf
import numpy as np

# # Add node demo
#
# x, y = mf.Input(), mf.Input()
# f1 = mf.Add(x, y)
# feed_dict = {x: 10, y: 5}
#
# sorted_nodes = mf.topological_sort(feed_dict)
# output = mf.forward_pass(f1, sorted_nodes)
#
# # NOTE: because topological_sort set the values for the `Input` nodes we
# # could # also access the value for x with x.value (same goes for y).
# print("{0} + {1} = {2} (according to miniflow)".format(feed_dict[x],
#                                                        feed_dict[y], output))

# # Mul node demo
#
# x1, x2, x3, x4, x5 = mf.Input(), mf.Input(), mf.Input(), mf.Input(), mf.Input()
# f2 = mf.Mul(x1, x2, x3, x4, x5)
# feed_dict_2 = {x1: 1, x2: 2, x3: 3, x4: 4, x5: 5}
#
# sorted_nodes_2 = mf.topological_sort(feed_dict_2)
# output_2 = mf.forward_pass(f2, sorted_nodes_2)
#
# print('5! = {0}'.format(output_2))

# # Linear node demo
#
# X, W, b = mf.Input(), mf.Input(), mf.Input()
#
# f = mf.Linear(X, W, b)
#
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])
#
# feed_dict = {X: X_, W: W_, b: b_}
#
# graph = mf.topological_sort(feed_dict)
# output = mf.forward_pass(f, graph)
#
# print(output) # should be 12.7 with this example

# Sigmoid demo

X, W, b = mf.Input(), mf.Input(), mf.Input()

f = mf.Linear(X, W, b)
g = mf.Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = mf.topological_sort(feed_dict)
output = mf.forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""

print(output)