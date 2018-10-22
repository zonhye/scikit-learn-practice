#!/usr/bin
# -*- coding: utf-8 -*-
#介绍数据集

from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
print(data.shape)
print(data[0:3])

digits = datasets.load_digits()
print(digits.images.shape)
plt.imshow(digits.images[-1], cmap = plt.cm.gray_r)
plt.show()
