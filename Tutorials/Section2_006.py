#!/usr/bin
# -*- coding: utf-8 -*-
#split方法分解数据集，以及直接使用cross_val_score进行交叉验证

from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits = 3)
for train_indices, test_indices in k_fold.split(X):
	print('Train: %s | test: %s' % (train_indices, test_indices))

svc = svm.SVC(C=1, kernel='linear')
for train, test in k_fold.split(X_digits):
	print(svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]))

print(cross_val_score(svc, X_digits, y_digits, cv = k_fold, n_jobs = -1))

print(cross_val_score(svc, X_digits, y_digits, cv = k_fold, scoring = 'precision_macro'))
