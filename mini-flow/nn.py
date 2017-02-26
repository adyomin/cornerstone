import numpy as np
import sys
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
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

# # Sigmoid demo
#
# X, W, b = mf.Input(), mf.Input(), mf.Input()
#
# f = mf.Linear(X, W, b)
# g = mf.Sigmoid(f)
#
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])
#
# feed_dict = {X: X_, W: W_, b: b_}
#
# graph = mf.topological_sort(feed_dict)
# output = mf.forward_pass(g, graph)
#
# """
# Output should be:
# [[  1.23394576e-04   9.82013790e-01]
#  [  1.23394576e-04   9.82013790e-01]]
# """
#
# print(output)

# # MSE demo
#
# y, a = mf.Input(), mf.Input()
# cost = mf.MSE(y, a)
#
# y_ = np.array([1, 2, 3])
# a_ = np.array([4.5, 5, 10])
#
# feed_dict = {y: y_, a: a_}
# graph = mf.topological_sort(feed_dict)
# # forward pass
# mf.forward_pass(graph)
#
# """
# Expected output
#
# 23.4166666667
# """
# print(cost.value)

# Backpropagation demo
#
# X, W, b = mf.Input(), mf.Input(), mf.Input()
# y = mf.Input()
# f = mf.Linear(X, W, b)
# a = mf.Sigmoid(f)
# cost = mf.MSE(y, a)
#
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2.], [3.]])
# b_ = np.array([-3.])
# y_ = np.array([1, 2])
#
# feed_dict = {X: X_, y: y_, W: W_, b: b_ }
#
# graph = mf.topological_sort(feed_dict)
# mf.forward_and_backward(graph)
# # return the gradients for each Input
# gradients = [t.gradients[t] for t in [X, y, W, b]]
#
# """
# Expected output
#
# [array([[ -3.34017280e-05,  -5.01025919e-05],
#        [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
#        [ 1.9999833]]), array([[  5.01028709e-05],
#        [  1.00205742e-04]]), array([ -5.01028709e-05])]
# """
# print(gradients)

"""
Check out the new network architecture and dataset!

Notice that the weights and biases are
generated randomly.

No need to change anything, but feel free to tweak
to test your network, play around with the epochs, batch size, etc!
"""

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = mf.Input(), mf.Input()
W1, b1 = mf.Input(), mf.Input()
W2, b2 = mf.Input(), mf.Input()

l1 = mf.Linear(X, W1, b1)
s1 = mf.Sigmoid(l1)
l2 = mf.Linear(s1, W2, b2)
cost = mf.MSE(y, l2)

feed_dict = {X: X_,
             y: y_,
             W1: W1_,
             b1: b1_,
             W2: W2_,
             b2: b2_
            }

epochs = 1024
# Total number of examples
m = X_.shape[0]
batch_size = 32
steps_per_epoch = m // batch_size

graph = mf.topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        mf.forward_and_backward(graph)

        # Step 3
        mf.sgd_update(trainables, learning_rate=0.01)

        loss += graph[-1].value

    sys.stdout.write("\rEpoch: {0}, Loss: {1:.4f}".format(i + 1,
                                                    loss/steps_per_epoch))
