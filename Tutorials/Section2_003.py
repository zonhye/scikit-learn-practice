#!/usr/bin
# -*- coding: utf-8 -*-
#线性回归

import numpy as np
from sklearn import datasets
from sklearn import linear_model

diabetes = datasets.load_diabetes()

#打印iris数据集的维度以及分类枚举值
print(diabetes.data.shape)
print(np.unique(diabetes.target))

#将数据集拆分为训练集和测试集
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

#实例化线性回归模型
regr = linear_model.LinearRegression()

#使用训练集训练先行回归模型
print(regr.fit(diabetes_X_train, diabetes_y_train))

#打印回归系数
print(regr.coef_)

#打印预测值与实际结果的均方误差
print(np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))

#打印模型的方差分数，1表示完美的预测，0表示X和y之间没有线性关系
print(regr.score(diabetes_X_test, diabetes_y_test))
