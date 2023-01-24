
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# Read real-estate data set
attr = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
data = pd.read_fwf('auto-mpg.data', names=attr)

# drop 'car name' column and remove missing values
data.drop('car name', axis=1, inplace=True)
data = data[data.horsepower != '?']
data.reset_index(drop=True, inplace=True)

# convert all columns to float64 data type
for i in data.columns:
    data[i] = data[i].astype('float64')

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=0)
Tree_model = DecisionTree(criterion='information_gain', max_depth=8)  # Split based on Information Gain

# fit the tree to the training data
Tree_model.fit(X_train, y_train)
y_hat = Tree_model.predict(X_test)
y1=np.array(y_hat)-np.array(y_test)
npY_hat = np.array(y_hat)
npY_hat = [ '%.1f' % elem for elem in npY_hat ]
npy_test = np.array(y_test)

# print the comparison of original output vs My Tree Prediction
print("Original Output vs My Tree Prediction")
for i in range(len(npY_hat)):
    print(npy_test[i],"          ",npY_hat[i])

# plot the decision tree
Tree_model.plot()
print('Root Mean Squared error: ', rmse(y_hat, y_test))
print('Mean Absolute Error: ', mae(y_hat, y_test))

# create another decision tree using DecisionTreeRegressor from sklearn
Sklearn_DecisionTree = DecisionTreeRegressor(max_depth=8)
Sklearn_DecisionTree.fit(X_train, y_train)      # fit the tree to the training data

Plot_Sk = tree.export_text(Sklearn_DecisionTree)
y_hat_new = Sklearn_DecisionTree.predict(X_test)
y_hat_new = pd.Series(y_hat_new)
y2 = np.array(y_hat_new)-np.array(y_hat)

npyy_hat = np.array(y_hat_new)
print("My Predicted Output vs SKlearn Predicted Output")
for i in range(len(npY_hat)):
    print(npY_hat[i],"          ",npyy_hat[i])

print('Root Mean Squared error: ', rmse(y_hat_new, y_test))
print('Mean Absolute Error: ', mae(y_hat_new, y_test))

