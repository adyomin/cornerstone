import numpy as np
import pandas as pd


batch_size = 5
num_targets = 10
targets = np.random.randint(0, 9, batch_size)
print('Labels: ')
print(targets)

print('Pandas solution: ')
print(pd.get_dummies(targets).values)

print('For loop solution: ')
option_1 = np.zeros((batch_size, num_targets))
for row, col in zip(option_1, targets):
    row[col] = 1
print(option_1)

print('numpy.ndarray.flat based solution: ')
# https://goo.gl/Yk7SXS
option_2 = np.zeros((batch_size, num_targets))
option_2.flat[np.arange(batch_size)*num_targets + targets] = 1
print(option_2)

print('Numpy indexing based solution: ')
# https://goo.gl/aD2hOA
option_3 = np.zeros(shape=(batch_size, num_targets))
option_3[np.arange(batch_size), targets] = 1
print(option_3)

