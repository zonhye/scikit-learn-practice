#!/usr/bin
# -*- coding: utf-8 -*-
#K近邻算法和维度灾难

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

#打印iris数据集的维度以及分类枚举值
print(iris.data.shape)
print(np.unique(iris_y))

#获取数据集下标的随机序列
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
print(indices)

#将数据集拆分为训练集和测试集
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

#实例化最近邻分类器
knn = KNeighborsClassifier()

#使用训练集训练最近邻分类器
print(knn.fit(iris_X_train, iris_y_train))

#使用训练好的分类器对测试集进行预测
print(knn.predict(iris_X_test))

#对比训练集的真实结果
print(iris_y_test)
