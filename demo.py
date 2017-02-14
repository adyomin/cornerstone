import network as net
import pandas as pd
import numpy as np


features = pd.read_csv('./data/bike_rental/features.csv')
targets = pd.read_csv('./data/bike_rental/targets.csv')

x = np.array(features.ix[:, 1:])
y = np.array(targets.ix[:, 2:])

size = 128
n = 512
np.random.seed(42)

test = net.Network(size=(56, 28, 14, 7, 2))
test.train(x, y, batch_size=size, eta=0.025, n_epochs=n)
