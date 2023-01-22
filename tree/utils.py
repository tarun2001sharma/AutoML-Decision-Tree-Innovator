import pandas as pd
import numpy as np
import math

def max_prob(Y):
    d={}
    for i in Y:
        if i not in d: d[i] = 1
        else: d[i] += 1
    return max(d, key= lambda x: d[x])


def calculate_mean(Y):
    if(len(Y)==0): return 0
    y=np.array(Y)
    return sum(Y)/len(Y)

def red_in_var(Y,attr):
    # Reduction in variance when discrete input and real output
    d={}
    for i in range(len(attr)):
        if attr[i] not in d: d[attr[i]] = [Y[i]]
        else: d[attr[i]].append(Y[i])
    red = np.var(Y)
    for i in d:
        red -= (len(d[i])/len(attr))*(np.var(d[i]))
    return red

def red_in_var_RI(Y, attr):
    # Reduction in Variance when real input and real output
    # returns (gain, split_value)
    if len(Y)==0:
        return [0,0]
    if(len(Y)==1):
        return [0,Y[0]]
    min_loss = 1e12
    Y = np.array(Y)
    attr = np.array(attr)

    Y = Y[attr.argsort()]
    attr = np.sort(attr)
 
    index = 0
    for i in range(len(attr)-1):
        a = np.var(Y[:i+1])
        b = np.var(Y[i+1:])

        if (a+b) < min_loss:
            index = i
            min_loss = (a+b)

    return [min_loss, (attr[index] + attr[index+1])/2]


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    Y = np.array(Y)
    _ , count_list = np.unique(Y, return_counts=True)
    prob_list = count_list/sum(count_list)
    entropy = 0
    for prob in prob_list:
        entropy += prob * math.log2(prob)
    return -entropy
    pass


def gini_index(Y: pd.Series, attr) -> float:
    """
    Function to calculate the gini index
    """
    d={}
    for i in range(len(attr)):
        if attr[i] not in d: d[attr[i]] = [Y[i]]
        else: d[attr[i]].append(Y[i])
    gini = 0
    for val in d.values():
        _ , count_list = np.unique(np.array(val), return_counts=True)
        prob_list = count_list/sum(count_list)
        prob_sq_list = np.square(prob_list)
        gini += (sum(count_list)/len(Y)) * (1-sum(prob_sq_list))

    return gini
    pass


def gini_index_RI(Y, attr):
    # Real Input and Discrete Output
    Y = np.array(Y)
    attr = np.array(attr)
    pairs = np.transpose((attr, Y))
    pairs_sorted = pairs[pairs[:, 0].argsort()]

    classes_list, count_list = np.unique(Y, return_counts=True)

    # List of 0s with length equal to number of classes in Y
    a = np.array([0 for x in range(len(classes_list))])
    # List of counts for each class; len(a) = len(b)
    b = count_list

    gini = 1e15

    # List of classes
    classes_list = list(classes_list)

    index = 0
    for i in range(len(attr)-1):

        # a contains counts in upper split and b in lower split
        a[classes_list.index(pairs_sorted[i][1])] += 1
        b[classes_list.index(pairs_sorted[i][1])] -= 1
        wt_a = sum(a)/(sum(a) + sum(b))
        wt_b = 1 - wt_a

        prob_a =[]
        for count in a:
            prob_a.append(count/sum(a))
        gini_upper = wt_a * (1-sum(np.square(np.array(prob_a))))    # Weighted Gini for upper split

        prob_b =[]
        for count in b:
            prob_b.append(count/sum(b))
        gini_lower = wt_b * (1-sum(np.square(np.array(prob_b))))   # Lower split
        p = gini_upper + gini_lower  # Weighted sum of gini
        
        # Minimize Gini
        if gini > p:
            index = i
            gini = p

    return [gini, (pairs[index][0] + pairs[index+1][0])/2]


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    value_entropy = 0
    class_list, count_list = np.unique(attr, return_counts=True)
    probability = count_list/sum(count_list)
    attr = np.array(attr)
    Y = np.array(Y)
    for i in range(len(class_list)):
        temp=[]
        for j in range(len(attr)):
            if class_list[i] == attr[j]:
                temp.append(Y[j])
        value_entropy += probability[i]*entropy(temp)
    return entropy(Y) - value_entropy
    pass

def information_gain_RI(Y, attr):
    # returns (gain, split)
    if(len(Y)==1):
        return [0,Y[0]]
    Y_temp = np.array(Y)
    attr = np.array(attr)
    pairs = np.transpose((attr, Y_temp))
    pairs_sorted = pairs[pairs[:, 0].argsort()]  # sort based on first column of real input

    class_list, count_list = np.unique(Y_temp, return_counts=True)
    class_list = list(class_list)
    a = [0 for x in range(len(class_list))]
    b = count_list
    min_entropy = 1e15
    index = 0
    for i in range(len(attr)-1):
        a[class_list.index(pairs_sorted[i][1])] += 1
        b[class_list.index(pairs_sorted[i][1])] -= 1

        wt_a = sum(a)/(sum(a) + sum(b))
        wt_b = 1 - wt_a
        prob_a = np.array(a)/sum(a)
        entropy_val = 0
        for prob in prob_a:
            if prob > 0:
                entropy_val += prob * math.log2(prob) * wt_a
        prob_b = np.array(b)/sum(b)
        for prob in prob_b:
            if prob > 0:
                entropy_val += prob * math.log2(prob) * wt_b

        if min_entropy>entropy_val:
            index = i
            min_entropy=entropy_val

    return [entropy(Y) - min_entropy, (pairs[index][0] + pairs[index+1][0])/2]
    pass
