#!/usr/bin
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

X = [[1, 2],[2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

###一维数组表示的多类分类
classif = OneVsRestClassifier(estimator = SVC(random_state = 0))
print(classif.fit(X, y).predict(X))

###二维数组表示的多类分类
y1 = LabelBinarizer().fit_transform(y)
print(classif.fit(X, y1).predict(X))

###二维数组表示的多标签分类
y2 = [[0, 1],[0, 2], [1, 3], [0, 2, 3], [2, 4]]
y2 = MultiLabelBinarizer().fit_transform(y2)
print(classif.fit(X,y2).predict(X))
