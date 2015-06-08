# -*- coding: utf-8 -*-

import os

import numpy as np
from sklearn.svm import LinearSVC

base = os.path.dirname(os.path.abspath(__file__))
auth = np.genfromtxt(os.path.join(base, 'CodeIQ_auth.txt'), delimiter=' ')

train_X = np.array([[x[0], x[1]] for x in auth])
labels = [int(x[2]) for x in auth]

test_X = np.genfromtxt(os.path.join(base, 'CodeIQ_mycoins.txt'), delimiter=' ')

clf = LinearSVC(C=1)
clf.fit(train_X, labels)

results = clf.predict(test_X)
for result, feature in zip(results, test_X):
    print(result, feature)

