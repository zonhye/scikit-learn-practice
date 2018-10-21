#!/usr/bin
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

# print(X)
# print(y)
# print(X_test)

clf = SVC()

###通过sklearn.pipeline.Pipeline.set_params方法更新估计器的参数
print(clf.set_params(kernel = 'linear').fit(X, y))
print("kernel设置为linear的预测结果：")
print(clf.predict(X_test))

print(clf.set_params(kernel = 'rbf').fit(X, y))
print("kernel设置为rbf的预测结果：")
print(clf.predict(X_test))
