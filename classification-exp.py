import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# generating dataset using sklearn
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

split = 0.7     # 70% of the data used for training purposes and the remaining 30% for test purposes

# train-test split
Xdata, ydata = X.copy(), y.copy()
X_train, y_train = Xdata[:int(split*len(Xdata))], ydata[:int(split*len(Xdata))]
X_test, y_test = Xdata[int(split*len(Xdata)):], ydata[int(split*len(Xdata)):]

tree = DecisionTree('information_gain', max_depth = 10)
tree.fit(pd.DataFrame(X_train),pd.Series(y_train))
y_predicted = tree.predict(pd.DataFrame(X_test))

print('Criteria :', 'information_gain')
print('Accuracy: ', accuracy(pd.Series(y_predicted), pd.Series(y_test)))
for cls in pd.Series(y).unique():
    print('Precision: ', precision(pd.Series(y_predicted), pd.Series(y_test), cls))
    print('Recall: ', recall(pd.Series(y_predicted), pd.Series(y_test), cls))


plt.scatter(X[:, 0], X[:, 1], c=y)


print()
print("------- 5-fold cross validation ------")

y_predicted = []
for i in range(4,-1,-1):
    left = int(len(Xdata)*(5-(i+1))/5)
    right = int((5-i)/5*len(Xdata))
    X_train, y_train = np.concatenate((Xdata[:left], Xdata[right:]), axis=0), np.concatenate((ydata[:left], ydata[right:]), axis=0)
    X_test, y_test = Xdata[left: right], y[left: right]
    tree = DecisionTree('information_gain', 10)
    tree.fit(pd.DataFrame(X_train), pd.Series(y_train))
    iter = tree.predict(pd.DataFrame(X_test))
    y_predicted += iter.tolist()


print('Criteria :', 'information_gain')
print('Accuracy: ', accuracy(pd.Series(y_predicted), pd.Series(y)))
for cls in pd.Series(y).unique():
    print('Precision: ', precision(pd.Series(y_predicted), pd.Series(y), cls))
    print('Recall: ', recall(pd.Series(y_predicted), pd.Series(y), cls))

print()
print("------ Nested cross validation ------")

cv_split = 0.8

optimum_depth = 0
max_accuracy = 0
Xdata = Xdata[:int(cv_split*len(Xdata))]
ydata = ydata[:int(cv_split*len(ydata))]

for i in range(tree.max_depth+1):
    y_predicted = []

    for k in range(5):
        left = int(len(Xdata)*k/5)
        right = int((k+1)/5*len(Xdata))
        X_train, y_train = np.concatenate((Xdata[:left], Xdata[right:]), axis=0), np.concatenate((ydata[:left], ydata[right:]), axis=0)
        X_test, y_test = Xdata[left: right], ydata[left: right]
        tree = DecisionTree('information_gain', i)
        tree.fit(pd.DataFrame(X_train), pd.Series(y_train))
        iter = tree.predict(pd.DataFrame(X_test))
        y_predicted += iter.tolist()

    accuray_at_depth = accuracy(pd.Series(y_predicted), pd.Series(ydata))
    print("Accuracy of tree at depth {0} = {1}".format(i,accuray_at_depth))
    if accuray_at_depth>max_accuracy:
        max_accuracy = accuray_at_depth
        optimum_depth = i

print("Optimum Tree Depth: ",optimum_depth)

