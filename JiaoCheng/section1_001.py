#!/usr/bin
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
import pickle

iris = datasets.load_iris()
digits = datasets.load_digits()

# print(digits.data)
# print(digits.target)
# print(digits.images[0])

clf = svm.SVC(gamma = 0.001, C = 100.)
print(clf.fit(digits.data[:-1], digits.target[:-1]))

print("实际结果：")
print(digits.images[-1])

print("预测结果：")
print(clf.predict(digits.data[-1:]))

###使用Python内置的持久化模块pickle将模型保存
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print("使用pickle保存的模型的预测结果：")
print(clf2.predict(digits.data[-1:]))

###使用sklearn中的joblib可以将模型存储到磁盘文件
joblib.dump(clf,'svc_clf.pkl')
clf3 = joblib.load('svc_clf.pkl')
print("使用joblib保存的模型的预测结果：")
print(clf3.predict(digits.data[-1:]))
