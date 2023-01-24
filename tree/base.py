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

class Node():
    def __init__(self):
        self.depth = 0
        self.node_attr = None
        self.values = []
        self.parent = None
        self.children = []
  

    def insert_child_attr(self, child):
        self.children.append(child)
        child.depth = self.depth + 1
        child.parent = self

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Node()
        pass

    def create_children(self, vals, node, input_type):
        if input_type=="discrete":
            for val in vals:                                       # creating children nodes based on the values of the best attr
                child = Node()
                node.values.append(val)
                node.insert_child_attr(child)                              # adding child and appending values together so that indexing is same for children and values
        else:
            for i in range(2):
                child = Node()
                node.insert_child_attr(child)
                node.values.append(vals)
        return node

    def child_subset_matrix(self, x, y, best_next_attr, value, input_type, i):
        X_new = x.copy(deep = True)                        # created copies for X and y
        Y_new = y.copy(deep = True)

        if input_type=="discrete":
            for idx,val in X_new[best_next_attr].iteritems():
                if value != val:
                    X_new = X_new.drop(idx)                    # create the children tables for each of the unique values of the optimum attr
                    Y_new = Y_new.drop(idx)            
            del X_new[best_next_attr]                         # remove the attr from that table as we cant use it again for splitting 
        else:
            for idx, val in X_new[best_next_attr].iteritems():
                if(i==0):
                    if val > value:
                        X_new = X_new.drop(idx)
                        Y_new = Y_new.drop(idx)
                else:
                    if val <= value:
                        X_new = X_new.drop(idx)
                        Y_new = Y_new.drop(idx)

        X_new = X_new.reset_index(drop=True)               # reset the indices 
        Y_new = Y_new.reset_index(drop=True)

        return X_new, Y_new


    def DIDO(self, x:pd.DataFrame, y:pd.Series, node):

        if(entropy(y) == 0):                     # If entropy is zero, no need to go for more depth   
            node.node_attr = y[0]                
            return

        if x.shape[1] == 0:                         # When we have used all the attributes
            Y_output = max_occurence(y)                   
            node.node_attr = Y_output
            return

        if (self.max_depth - node.depth >= 1) and x.shape[1] != 0:

            attr_list = x.columns.tolist()      # attr_list to store the feature column names 
            best_next_attr = attr_list[0]      # best attribute to make the decision
            max_gain = 0                        # Information gain criteria
            mini_gini = 1e15                    # Gini Index criteria

            for attr in attr_list:            
                if self.criterion == 'information_gain':
                    gain = information_gain(y, x[attr])
                    if gain >= max_gain:
                        max_gain = gain
                        best_next_attr = attr

                if self.criterion == 'gini_index':
                    gini = gini_index(y, x[attr])
                    if gini<mini_gini:
                        mini_gini = gini
                        best_next_attr = attr
        
            node.node_attr = best_next_attr                    # the selected best attribute becomes the node
            vals = np.unique(np.array(x[best_next_attr]))         # finding the values of the best attr

            node = self.create_children(vals, node, "discrete")
            
            for value in node.values:
                X_new, Y_new = self.child_subset_matrix(x, y, best_next_attr, value, "discrete", i = 0)

                if(self.max_depth - node.depth == 1):              # one level before the max_depth, we simply use the most probable output value
                    Y_output = max_occurence(pd.Series(Y_new.tolist()))
                    node.children[node.values.index(value)].node_attr = Y_output
                else:
                    self.DIDO(X_new, Y_new, node.children[node.values.index(value)]) # else we use recursion to create the tree for more depth

        pass

    def DIRO(self, x:pd.DataFrame, y:pd.Series, node):

        if(len(y) == 1):                # If only one example remains in the table
            node.node_attr = y[0]
            return

        if x.shape[1] == 0:
            Y_output = meanY(y)
            node.node_attr = Y_output
            return

        if (self.max_depth - node.depth >= 1) and x.shape[1] != 0:
            max_gain = -1e15
            attr_list = x.columns.tolist() 
            best_next_attr = attr_list[0]   
            for attr in attr_list:
                gain = red_in_var(y, x[attr])

                if (gain >= max_gain):
                    max_gain = gain
                    best_next_attr = attr

            vals = np.unique(np.array(x[best_next_attr]))         
            node.node_attr = best_next_attr                 

            node = self.create_children(vals, node, "discrete")

            for value in node.values:
                X_new, Y_new = self.child_subset_matrix(x, y, best_next_attr, value, "discrete", i = 0)
                if(self.max_depth - node.depth == 1):
                    Y_output = meanY(pd.Series(Y_new.tolist()))                       
                    node.children[node.values.index(value)].node_attr = Y_output        
                else:
                    self.DIRO(X_new, Y_new, node.children[node.values.index(value)])    

        pass


    def RIDO(self, x:pd.DataFrame, y:pd.Series, node):
        
        if(entropy(y) == 0):
            node.node_attr = y[0]
            return

        if x.shape[1] == 0:                         
            Y_output = max_occurence(y)                  
            node.node_attr = Y_output
            return

        if (self.max_depth - node.depth >= 1) and x.shape[1] != 0:
            max_gain = -1e15   
            mini_gini = 1e15
            split_val = 0
            attr_list = x.columns.tolist()
            best_next_attr = attr_list[0]
            for attr in attr_list:
                if(self.criterion == 'information_gain'):
                    output = information_gain_real_in(y,  x[attr])
                    gain = output[0]
                    if (gain >= max_gain):
                        max_gain = gain
                        split_val = output[1]
                        best_next_attr = attr

                if(self.criterion == 'gini_index'):
                    output = gini_index_real_in(y, x[attr])
                    gini = output[0]
                    if (gini <= mini_gini):
                        mini_gini = gini
                        split_val = output[1]
                        best_next_attr = attr

            node.node_attr = best_next_attr

            node = self.create_children(split_val, node, "real")

            for i in range(2):

                X_new, Y_new = self.child_subset_matrix(x, y, best_next_attr, split_val, "real", i)                
                if(self.max_depth - node.depth == 1):
                    node.children[i].node_attr = max_occurence(pd.Series(Y_new.tolist()))
                else:
                    self.RIDO(X_new, Y_new, node.children[i])

        pass


    def RIRO(self, x: pd.DataFrame, y: pd.Series, node):

        if(len(y) == 1):
            node.node_attr = y[0]
            return
        if x.shape[1] == 0:
            Y_output = meanY(y)
            node.node_attr = Y_output
            return

        if (self.max_depth - node.depth >= 1) and x.shape[1] != 0:
            max_gain = -1e15
            split_val = 0
            attr_list = x.columns.tolist()
            best_next_attr = attr_list[0]
            for attr in attr_list:
                output = red_in_var_real_in(y, x[attr])
                gain = output[0]
                if (gain >= max_gain):
                    max_gain = gain
                    split_val = output[1]
                    best_next_attr = attr


            node.node_attr = best_next_attr

            node = self.create_children(split_val, node, "real")

            for i in range(2):
                X_new, Y_new = self.child_subset_matrix(x, y, best_next_attr, split_val, "real", i)                
                if(self.max_depth - node.depth == 1):
                    node.children[i].node_attr = meanY(pd.Series(Y_new.tolist()))
                else:
                    self.RIRO(X_new, Y_new, node.children[i])


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # Check the type of input and output, and call corresponding function to create tree
        if isinstance(X.iloc[0][0],(int, np.int64)) and isinstance(y[0],(int, np.int64)):
            if(self.max_depth == 0):    
                Y_output = max_occurence(y)
                self.root.node_attr = Y_output
            else:
                self.DIDO(X, y, self.root)
        elif isinstance(X.iloc[0][0], (float, np.float64)) and isinstance(y[0], (int, np.int64)):
            if(self.max_depth == 0):                       
                Y_output = max_occurence(y)
                self.root.node_attr = Y_output
            else:
                self.RIDO(X, y, self.root)
        elif isinstance(X.iloc[0][0], (int, np.int64)) and isinstance(y[0], (float, np.float64)):
            if(self.max_depth == 0):
                Y_output = meanY(y)
                self.root.node_attr = Y_output
            else:
                self.DIRO(X, y, self.root)
        else:
            if(self.max_depth == 0):
                Y_output = meanY(y)
                self.root.node_attr = Y_output
            else:
                self.RIRO(X, y, self.root)
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        y=[]

        # For discrete input, we traverse the tree if node value matches attr name
        if isinstance(X.iloc[0][0], (int, np.integer)):
            for i in range (X.shape[0]):
                curr_example = X.iloc[i]
                root_node = self.root
                while (len(root_node.children)>0):
                    value = curr_example[root_node.node_attr]
                    root_node = root_node.children[root_node.values.index(value)]
                y.append(root_node.node_attr)
        
        # For real input, we traverse using splits
        else:
            for i in range(X.shape[0]):
                curr_example = X.iloc[i]
                root_node = self.root
                while (len(root_node.children)>0):
                    value = curr_example[root_node.node_attr]
                    if value <= root_node.values[0]:
                        root_node = root_node.children[0]
                    else:
                        root_node = root_node.children[1]
                y.append(root_node.node_attr)

        y = pd.Series(y)
        return y
        pass

    def print_tree_discrete_in(self, node):
            if len(node.children) == 0:
                print(" | "*node.depth, " --> ", "Value = ", str(node.node_attr), ", Depth = " , node.depth)
            else:
                count = 0
                for i in node.children:
                    print(" | "*node.depth, "?(X" + str(node.depth + 1) + " = " + str(node.values[count]) + ")")
                    self.print_tree_discrete_in(i)
                    count += 1

    def print_tree_real_in(self, node):
            if len(node.children) == 0:
                print(" | "*node.depth, " --> ", "Value = ", str(node.node_attr), ", Depth = " , node.depth)
            else:
                count = 0
                for i in node.children:
                    if count == 0:
                        print(" | "*node.depth, "?(X" + str(node.depth + 1) + " < " + str(node.values[count]) + ")")
                        self.print_tree_real_in(i)
                    else:
                        print(" | "*node.depth, "?(X" + str(node.depth + 1) + " > " + str(node.values[count]) + ")")
                        self.print_tree_real_in(i)
                    count += 1

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

        root_node = self.root
        if isinstance(self.root.values[0], (int, np.integer)):
            self.print_tree_discrete_in(root_node)
        else:
            self.print_tree_real_in(root_node)
        pass
