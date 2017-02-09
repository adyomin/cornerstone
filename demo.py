import network as net
import pandas as pd
import numpy as np


features = pd.read_csv('~/Yandex.Disk.localized/Projects'
                       '/deep_learning_program/week1/project_1/features.csv')
targets = pd.read_csv('~/Yandex.Disk.localized/Projects'
                      '/deep_learning_program/week1/project_1/targets.csv')

x = np.array(features.ix[:, 1:])
y = np.array(targets.ix[:, 2:])

size = 512
n = 128
np.random.seed(42)

test = net.Network(features=x, targets=y, h_size=(112, 28), eta=0.001)
test.train(batch_size=size, n_epochs=n)
