#!/usr/bin
# -*- coding: utf-8 -*-
#收缩回归系数到0，岭回归

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0, 2].T

#选择线性回归模型或者岭回归模型，对岭回归模型，岭参数alpha越大，偏差越大，方差越小
#regr = linear_model.LinearRegression()
regr = linear_model.Ridge(alpha = .1)
plt.figure()

np.random.seed(0)
for _ in range(6):
	this_X = .1*np.random.normal(size=(2, 1)) + X
	regr.fit(this_X, y)
	plt.plot(test, regr.predict(test))
	plt.scatter(this_X, y, s=3)

plt.show()
