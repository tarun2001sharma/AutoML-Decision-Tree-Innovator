"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class Treenode():
    def __init__(self):
        self.feature_name = None
        self.parent = None
        self.level = 0
        self.values = []
        self.children = []
  

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        child.level = self.level + 1

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Treenode()
        pass

    def DIDO_fit(self, x:pd.DataFrame, y:pd.Series, node):

        if(self.max_depth == 0):                    # if depth of tree is zero simply give maximum outputted class    
            Y_output = max_prob(y)
            node.feature_name = Y_output

        if(entropy(y) == 0):                        # Checks if we get final result on the current level
            node.feature_name = y[0]                # Assigns feature to the node

        if x.shape[1] == 0:                         # check number of columns
            Y_output = max_prob(y)                   # for the case where there are repeated entries with different output
            node.feature_name = Y_output

        if (self.max_depth - node.level >= 1) and x.shape[1] != 0:

            feature_names = x.columns.tolist()      # feature_names stores the column headers
            optimum_feature = feature_names[0]      # finding the best feature to split on
            maximum_gain = 0
            minimum_gini = 1e10

            for feature in feature_names:            
                if self.criterion == 'information_gain':
                    gain = information_gain(y, x[feature])

                    if gain >= maximum_gain:
                        maximum_gain = gain
                        optimum_feature = feature

                if self.criterion == 'gini_index':
                    gini = gini_index(y, x[feature])

                    if gini<minimum_gini:
                        minimum_gini = gini
                        optimum_feature = feature
        
            node.feature_name = optimum_feature                    # assigning the name of the node of the tree as the feature name
            vals = np.unique(np.array(x[optimum_feature]))         # creating a list of unique values of the best attribute

            for val in vals:                                       # creating children nodes based on the values of the best feature
                child = Treenode()
                node.values.append(val)
                node.add_child(child)                              # adding child and appending values together so that indexing is same for children and values

            for value in node.values:
                X_new = x.copy(deep = True)                        # created copies for X and y
                Y_new = y.copy(deep = True)

                for idx,val in X_new[optimum_feature].iteritems():
                    if value != val:
                        X_new = X_new.drop(idx)                    # create the children tables for each of the unique values of the optimum feature
                        Y_new = Y_new.drop(idx)

                del X_new[optimum_feature]                         # remove the feature from that table as we cant use it again for splitting 
                X_new = X_new.reset_index(drop=True)               # reset the indices 
                Y_new = Y_new.reset_index(drop=True)                
                if(self.max_depth - node.level == 1):              # if we are only one level above the final decision 
                    Y_output = max_prob(pd.Series(Y_new.tolist()))  # give result using probability (returns the maximum outputted class at this point)
                    node.children[node.values.index(value)].feature_name = Y_output
                else:
                    self.DIDO_fit(X_new, Y_new, node.children[node.values.index(value)]) # recurse for children

        pass

    def DIRO_fit(self, x:pd.DataFrame, y:pd.Series, node):

        if(self.max_depth == 0):                 # if depth of tree is zero simply give mean of outputs
            Y_output = calculate_mean(y)
            node.feature_name = Y_output

        if x.shape[1] == 0:
            Y_output = calculate_mean(y)           # for the case where there are repeated entries with different output
            node.feature_name = Y_output

        if (self.max_depth - node.level >= 1) and x.shape[1] != 0:
            maximum_gain = -1e10
            feature_names = x.columns.tolist()   # feature_names stores the column headers
            optimum_feature = feature_names[0]   # finding the best feature to split on
            for feature in feature_names:
                gain = red_in_var(y, x[feature])

                if (gain >= maximum_gain):
                    maximum_gain = gain
                    optimum_feature = feature

            vals = np.unique(np.array(x[optimum_feature]))         # creating a list of unique values of the best attribute
            node.feature_name = optimum_feature                    # assigning the name of the node of the tree as the feature name

            for val in vals:                                       # creating children nodes based on the values of the best feature 
                child = Treenode()
                node.add_child(child)
                node.values.append(val)

            for value in node.values:
                X_new = x.copy(deep = True)
                Y_new = y.copy(deep = True)

                for idx,val in X_new[optimum_feature].iteritems():
                    if value != val:
                        X_new = X_new.drop(idx)
                        Y_new = Y_new.drop(idx)

                del X_new[optimum_feature]                      # remove the feature from that table as we cant use it again for splitting 
                X_new = X_new.reset_index(drop=True)            # reset the indices
                Y_new = Y_new.reset_index(drop=True)
                if(self.max_depth - node.level == 1):
                    Y_output = calculate_mean(pd.Series(Y_new.tolist()))                      # if we are only one level above the final decision 
                    node.children[node.values.index(value)].feature_name = Y_output         # give result using probability (returns the maximum outputted class at this point)
                else:
                    self.DIRO_fit(X_new, Y_new, node.children[node.values.index(value)])    # recurse for children

        pass


    def RIDO_fit(self, x:pd.DataFrame, y:pd.Series, node):

        if(self.max_depth == 0):
            Y_output = max_prob(y)
            node.feature_name = Y_output
        
        if(len(y) == 1):                                     
        
            node.feature_name = y[0]                          
            return

        if (self.max_depth - node.level >= 1) and x.shape[1] != 0:
            maximum_gain = -1e10   
            split_val = 0
            feature_names = x.columns.tolist()
            optimum_feature = feature_names[0]
            for feature in feature_names:
                if(self.criterion == 'information_gain'):
                    output = information_gain_RI(y,  x[feature])
                    gain = output[0]
                    if (gain >= maximum_gain):
                        maximum_gain = gain
                        split_val = output[1]
                        optimum_feature = feature

                if(self.criterion == 'gini_index'):
                    output = gini_index_RI(y, x[feature])
                    gain = -output[0]
                    if (gain >= maximum_gain):
                        maximum_gain = gain
                        split_val = output[1]
                        optimum_feature = feature

            node.feature_name = optimum_feature

            for i in range(2):
                child = Treenode()
                node.add_child(child)
                node.values.append(split_val)


            for i in range(2):
                X_new = x.copy(deep = True)
                Y_new = y.copy(deep = True)

                for idx, val in X_new[optimum_feature].iteritems():
                    if(i==0):
                        if val > split_val:
                            X_new = X_new.drop(idx)
                            Y_new = Y_new.drop(idx)
                    else:
                        if val <= split_val:
                            X_new = X_new.drop(idx)
                            Y_new = Y_new.drop(idx)

                X_new = X_new.reset_index(drop=True)
                Y_new = Y_new.reset_index(drop=True)
                
                if(self.max_depth - node.level == 1):
                    node.children[i].feature_name = max_prob(pd.Series(Y_new.tolist()))
                else:
                    self.RIDO_fit(X_new, Y_new, node.children[i])

        pass


    def RIRO_fit(self, x: pd.DataFrame, y: pd.Series, node):

        if(self.max_depth == 0):
            Y_output = calculate_mean(y)
            node.feature_name = Y_output

        if(len(y) == 1):
            node.feature_name = y[0]
            return
        if x.shape[1] == 0:
            Y_output = calculate_mean(y)
            node.feature_name = Y_output

        if (self.max_depth - node.level >= 1) and x.shape[1] != 0:
            maximum_gain = -1e10
            split_val = 0
            feature_names = x.columns.tolist()
            optimum_feature = feature_names[0]
            for feature in feature_names:
                output = red_in_var_RI(y, x[feature])
                gain = output[0]
                if (gain >= maximum_gain):
                    maximum_gain = gain
                    split_val = output[1]
                    optimum_feature = feature


            node.feature_name = optimum_feature

            for i in range(2):
                child = Treenode()
                node.add_child(child)
                node.values.append(split_val)


            for i in range(2):
                X_new = x.copy(deep=True)
                Y_new = y.copy(deep=True)

                for idx, val in X_new[optimum_feature].iteritems():
                    if i == 0:
                        if val > split_val:
                            X_new = X_new.drop(idx)
                            Y_new = Y_new.drop(idx)
                    else:
                        if val <= split_val:
                            X_new = X_new.drop(idx)
                            Y_new = Y_new.drop(idx)


                X_new = X_new.reset_index(drop=True)
                Y_new = Y_new.reset_index(drop=True)

                if(self.max_depth - node.level == 1):
                    node.children[i].feature_name = calculate_mean(pd.Series(Y_new.tolist()))
                else:
                    self.RIRO_fit(X_new, Y_new, node.children[i])


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        if isinstance(X.iloc[0][0],(int, np.int64)) and isinstance(y[0],(int, np.int64)):
            self.DIDO_fit(X, y, self.root)
        elif isinstance(X.iloc[0][0], (int, np.int64)) and isinstance(y[0], (float, np.float64)):
            self.DIRO_fit(X, y, self.root)
        elif isinstance(X.iloc[0][0], (float, np.float64)) and isinstance(y[0], (float, np.float64)):
            self.RIRO_fit(X, y, self.root)
        else:
            self.RIDO_fit(X, y, self.root)

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        y=[]
        if isinstance(X.iloc[0][0], (int, np.integer)):
            for i in range (X.shape[0]):
                row = X.iloc[i]
                tree_root = self.root
                while (len(tree_root.children)!=0):
                    value = row[tree_root.feature_name]
                    tree_root = tree_root.children[tree_root.values.index(value)]
                y.append(tree_root.feature_name)
            
        else:
            for i in range(X.shape[0]):
                row = X.iloc[i]
                tree_root = self.root
                while (len(tree_root.children) != 0):
                    value = row[tree_root.feature_name]
                    if value <= tree_root.values[0]:
                        tree_root = tree_root.children[0]
                    else:
                        tree_root = tree_root.children[1]
                y.append(tree_root.feature_name)

        return pd.Series(y)
        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
