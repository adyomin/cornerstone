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