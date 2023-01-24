
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Read real-estate data set

attr = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]

data = pd.read_fwf('auto-mpg.data', names=attr)
print(data)
data.drop('car name', axis=1, inplace=True)
data = data[data.horsepower != '?']
data.reset_index(drop=True, inplace=True)

for i in data.columns:
    data[i] = data[i].astype('float64')

# data.drop(data.index[data['horsepower'] == "?"], inplace=True)
# print(data)

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=0)
My_Tree = DecisionTree(criterion='information_gain', max_depth=8)  # Split based on Inf. Gain
My_Tree.fit(X_train, y_train)
y_hat = My_Tree.predict(X_test)
y1=np.array(y_hat)-np.array(y_test)
npY_hat = np.array(y_hat)
npY_hat = [ '%.1f' % elem for elem in npY_hat ]
npy_test = np.array(y_test)
print("Original Output vs My Tree Prediction")
for i in range(len(npY_hat)):
    print(npy_test[i],"          ",npY_hat[i])

My_Tree.plot()

print('RMSE: ', rmse(y_hat, y_test))
print('MAE: ', mae(y_hat, y_test))


Sk_Tree = DecisionTreeRegressor(max_depth=8)
Sk_Tree.fit(X_train, y_train)


Plot_Sk = tree.export_text(Sk_Tree)
y_hat_Sk = Sk_Tree.predict(X_test)
y_hat_Sk = pd.Series(y_hat_Sk)
y2 = np.array(y_hat_Sk)-np.array(y_hat)

npyy_hat = np.array(y_hat_Sk)
print("My Predicted Output vs SKlearn Predicted Output")
for i in range(len(npY_hat)):
    print(npY_hat[i],"          ",npyy_hat[i])

print('RMSE: ', rmse(y_hat_Sk, y_test))
print('MAE: ', mae(y_hat_Sk, y_test))

