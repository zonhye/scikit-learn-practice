#!/usr/bin
# -*- coding: utf-8 -*-
#网格搜索

from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = svm.SVC(kernel = 'linear')

Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator = svc, param_grid = dict(C=Cs), n_jobs = -1)
clf.fit(X_digits[:1000], y_digits[:1000])

print(clf.best_score_)

print(clf.best_estimator_.C)

print(clf.score(X_digits[1000:], y_digits[1000:]))
