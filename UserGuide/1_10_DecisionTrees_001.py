#!/usr/bin
# -*- coding: utf-8 -*-
#构造决策树分类器，并使用graphviz导出图形化的决策树

import imp
import graphviz
from sklearn import tree
from sklearn.datasets import load_iris

X = [[0, 0], [1, 1]]
y = [0, 1]

#构造一个简单的决策树分类器，并进行训练
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

#预测类别
print(clf.predict([[2., 2.]]))
#预测每个类的概率
print(clf.predict_proba([[2., 2.]]))

#装载iris数据集并训练决策树分类器
iris = load_iris()
clf_iris = tree.DecisionTreeClassifier()
clf_iris.fit(iris.data, iris.target)

#使用export_graphviz导出器，将训练好的决策树以Graphviz格式导出
# dot_data = tree.export_graphviz(clf_iris, out_file = None)
dot_data = tree.export_graphviz(clf_iris, out_file = None, \
								feature_names = iris.feature_names, \
								class_names = iris.target_names, \
								filled = True, rounded = True, \
								special_characters = True)

#将Graphviz格式的决策树导出到pdf文档
graph = graphviz.Source(dot_data)
graph.render("iris")

#使用训练好的决策时分类器进行预测
print(clf_iris.predict(iris.data[:1, :]))
print(clf_iris.predict_proba(iris.data[:1, :]))
