from mini_flow import miniflow as mf

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

# Linear node demo

inputs, weights, bias = mf.Input(), mf.Input(), mf.Input()

f = mf.Linear(inputs, weights, bias)

feed_dict = { inputs: [6, 14, 3],
              weights: [0.5, 0.25, 1.4],
              bias: 2
              }

graph = mf.topological_sort(feed_dict)
output = mf.forward_pass(f, graph)

print(output) # should be 12.7 with this example