# based on @iamtrask materials for Udacity.

import network as nn
from collections import Counter
import numpy as np
import re

f = open('./data/reviews.txt','r')
reviews = [line.strip('\n') for line in f]
f.close()

f = open('./data/labels.txt','r')
text_labels = [line.strip('\n').upper() for line in f]
f.close()

f = open('./data/full_stop_list.txt','r')
full_stop_list = [line.strip('\n') for line in f]
f.close()

total_counts = Counter()

for review in reviews:
        total_counts.update(re.findall('\w{3,}', review.lower()))

for word in full_stop_list:
    del total_counts[word]

vocab = set(total_counts.keys())

word2index = Counter()

for i, word in enumerate(vocab):
    word2index[word] = i

limit = 1000

features = np.zeros((limit, len(vocab)))
for i, review in enumerate(reviews[:limit]):
    for word in review.split():
        if word in vocab:
            features[i, word2index[word]] += 1

labels = np.zeros(shape=(limit, 1))
for i, label in enumerate(text_labels[:limit]):
    if label == 'POSITIVE':
        labels[i, 0] = 1
    else:
        labels[i, 0] = 0

nn_model = nn.Network((73297, 256, 64, 1))
nn_model.train(features, labels, batch_size=16, eta=0.01, n_epochs=100)
