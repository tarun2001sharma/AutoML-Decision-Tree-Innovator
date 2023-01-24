
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from datetime import datetime
import os
from tqdm import tqdm

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

def create_dataset(type = "riro", N = 30, M = 5):

    num_classes = 2     #   binary

    if type == "riro":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    elif type == "rido":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(num_classes, size = N), dtype="category")
    elif type == "dido":
        X = pd.DataFrame({i: pd.Series(np.random.randint(
            num_classes, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(num_classes, size=N), dtype="category")
    else:
        X = pd.DataFrame({i: pd.Series(np.random.randint(
            num_classes, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))

    return X, y

def determine_time_M(M):
    dict = {"riro": {"learn":[], "predict":[]} , "rido": {"learn":[], "predict":[]} , "dido": {"learn":[], "predict":[]} , "diro": {"learn":[], "predict":[]} }

    for key in dict.keys():
        for m in tqdm(range(1, M)):
            X, y = create_dataset(key, 30, m)
            tree = DecisionTree()
            
            # print(X, y)

            start = datetime.now()
            tree.fit(X, y)
            end = datetime.now()
            learning_time = (end-start).total_seconds()
            dict[key]["learn"].append(learning_time)

            start = datetime.now()
            tree.predict(X)
            end = datetime.now()
            prediction_time = (end-start).total_seconds()
            dict[key]["predict"].append(prediction_time)

    return dict

def determine_time_N(N):
    dict = {"riro": {"learn":[], "predict":[]} , "rido": {"learn":[], "predict":[]} , "dido": {"learn":[], "predict":[]} , "diro": {"learn":[], "predict":[]} }

    for key in dict.keys():
        for n in tqdm(range(1, N)):
            X, y = create_dataset(key, n, 5)
            tree = DecisionTree()

            start = datetime.now()
            tree.fit(X, y)
            end = datetime.now()
            learning_time = (end-start).total_seconds()
            dict[key]["learn"].append(learning_time)

            start = datetime.now()
            tree.predict(X)
            end = datetime.now()
            prediction_time = (end-start).total_seconds()
            dict[key]["predict"].append(prediction_time)

    return dict

def plot(dict_M, dict_N):

    plt.figure()
    for key in dict_M.keys():
        # print(dict_M[key])
        # plt.plot(dict_M[key]["learn"], list(range(len(dict_M[key]["learn"]))), label = key)
        plt.plot(dict_M[key]["learn"], label = key)
    plt.xlabel("M")
    plt.ylabel("time")
    plt.title("Learning tree vs M")
    plt.legend()
    plt.savefig("learn_M.png")

    plt.figure()
    for key in dict_M.keys():
        plt.plot(dict_M[key]["predict"], label = key)
    plt.xlabel("M")
    plt.ylabel("time")
    plt.title("Prediction on test vs M")
    plt.legend()
    plt.savefig("predict_M.png")

    plt.figure()
    for key in dict_N.keys():
        plt.plot(dict_N[key]["learn"], label = key)
    plt.xlabel("N")
    plt.ylabel("time")
    plt.title("Learning tree vs N")
    plt.legend()
    plt.savefig("learn_N.png")

    plt.figure()
    for key in dict_N.keys():
        plt.plot(dict_N[key]["predict"], label = key)
    plt.xlabel("N")
    plt.ylabel("time")
    plt.title("Prediction on test vs N")
    plt.legend()
    plt.savefig("predict_N.png")


dict_M = determine_time_M(15)
dict_N = determine_time_N(100)

# print(dict_M)
# print(dict_N)

plot(dict_M, dict_N)
