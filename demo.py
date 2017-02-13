import network as net
import pandas as pd
import numpy as np


features = pd.read_csv('.data/features.csv')
targets = pd.read_csv('.data/targets.csv')

x = np.array(features.ix[:, 1:])
y = np.array(targets.ix[:, 2:])

size = 512
n = 512
np.random.seed(42)

test = net.Network(features=x, targets=y, h_size=(56, 28, 14, 7), eta=0.005)
test.train(batch_size=size, n_epochs=n)

#pushing from PyCharm using token thingie 2