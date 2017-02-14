import network as net
import pandas as pd
import numpy as np


train_features = pd.read_csv('./data/train_features.csv')
train_targets = pd.read_csv('./data/train_targets.csv')
val_features = pd.read_csv('./data/val_features.csv')
val_targets = pd.read_csv('./data/val_targets.csv')

x_train = np.array(train_features.ix[:, 1:])
y_train = np.array(train_targets.ix[:, 2:])
x_val = np.array(val_features.ix[:, 1:])
y_val = np.array(val_targets.ix[:, 2:])

size = 128
n = 512
np.random.seed(42)

test = net.Network(size=(56, 28, 14, 7, 2))
test.train(x_train, y_train, batch_size=size, eta=0.025, n_epochs=n)
