# based on @iamtrask materials for Udacity.

import network as nn
from collections import Counter
import numpy as np
import re
import sys

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


batch_size = 512
eta = 0.01
n_records = len(text_labels)
n_features = len(vocab)
features = np.zeros(shape=(batch_size, n_features))
labels = np.zeros(shape=(batch_size, 1))
nn_model = nn.Network((73297, 1024, 1))

for epoch in range(5):
    for i in range(0, n_records, batch_size):
        features *= 0
        labels *= 0
        for j, review in enumerate(reviews[i:i + batch_size]):
            for word in review.split():
                if word in vocab:
                    features[j, word2index[word]] += 1
        for k, label in enumerate(text_labels[i:i + batch_size]):
            if label == 'POSITIVE':
                labels[k, 0] = 1
            else:
                labels[k, 0] = 0
        nn_model.train_single_loop(features, labels, batch_size, eta)
        percentage = epoch/10*100
        t_loss = nn_model.evaluate(x=features, y=labels)
        print('Progress: {0:.2f}%, Training loss: {1:.4f}'.format(percentage,
                                                              t_loss))
    # sys.stdout.write('\rProgress: {0:.2f}% ... Training loss: '
    #              '{1:.4f}'.format(percentage, t_loss))
