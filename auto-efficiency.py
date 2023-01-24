
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read real-estate data set

attr = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]

data = pd.read_fwf('auto-mpg.data', names=attr)
print(data)
data.drop('car name', axis=1, inplace=True)
data = data[data.horsepower != '?']
# data.drop(data.index[data['horsepower'] == "?"], inplace=True)
print(data)


