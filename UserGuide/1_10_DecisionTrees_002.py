#!/usr/bin
# -*- coding: utf-8 -*-
#使用matplotlib直观展现决策树分类器的预测结果和训练样本集

import imp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

#设置参数
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

#装载iris数据集
iris = load_iris()

#将4个属性，排列组合成6个属性对，分别训练决策树分类器，并通过matplotlib画图直观展示
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
	X = iris.data[:, pair]
	y = iris.target

	clf = DecisionTreeClassifier().fit(X, y)

	plt.subplot(2, 3, pairidx + 1)

	#获取训练集上属性对的最大值，最小值
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx,yy = np.meshgrid(np.arange(x_min, x_max, plot_step), \
						np.arange(y_min, y_max, plot_step))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	#通过画等高线的函数，在两个属性的最大值最小值范围内，展示预测结果
	cs = plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)

	plt.xlabel(iris.feature_names[pair[0]])
	plt.ylabel(iris.feature_names[pair[1]])
	plt.axis("tight")

	#画出训练样本集
	for i, color in zip(range(n_classes), plot_colors):
		idx = np.where(y == i)
		plt.scatter(X[idx, 0], X[idx, 1], c = color, label = iris.target_names[i], \
					cmap = plt.cm.Paired)
	plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()
