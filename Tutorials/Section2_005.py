#!/usr/bin
# -*- coding: utf-8 -*-
#score方法可以在新的数据集上判断拟合（或预测）质量

from sklearn import datasets, svm
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

#score方法可以在新的数据集上判断拟合（或预测）质量，越大越好
svc = svm.SVC(C=1, kernel='linear')
print(svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:]))

#可以通过折叠数据进行连续训练和测试
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)

scores = list()
for k in range(3):
	X_train = list(X_folds)
	X_test = X_train.pop(k)
	X_train = np.concatenate(X_train)
	y_train = list(y_folds)
	y_test = y_train.pop(k)
	y_train = np.concatenate(y_train)
	scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

print(scores)
