import pandas as pd
import numpy as np
import math

def max_prob(y):
    dict={}
    for i in y:
        if i not in dict:
            dict[i] = 1
        else:
            dict[i] += 1
    return max(dict, key= lambda x: dict[x])


def calculate_mean(y):
    if(len(y)==0):
        return 0
    y=np.array(y)
    return sum(y)/len(y)

def red_in_var(Y,attr):

    # Reduction in variance when discrete input and real output

    D={}
    for i in range(len(attr)):
        if attr[i] not in D:
            D[attr[i]] = [Y[i]]
        else:
            D[attr[i]].append(Y[i])
    A = np.var(Y)
    for i in D:
        A -= (len(D[i])/len(attr))*(np.var(D[i]))

    return A
 


def red_in_var_RI(Y, attr):

    # Reduction in Variance when real input and real output
    # returns (gain, split_value)

    if len(Y)==0:
        return [0,0]
    if(len(Y) == 1):
        return [0, Y[0]]
    min_loss = 1e10
    # A = np.var(Y)
    Y = np.array(Y)
    attr = np.array(attr)
    # combined = np.transpose((attr, Y))  # pairs
    # X = combined[combined[:, 0].argsort()]
    # print(combined)

    Y = Y[attr.argsort()]
    attr = np.sort(attr)
 
    idx = 0
    for i in range(len(attr)-1):
        # a = combined[1][:i+1]
        # b = combined[1][i+1:]
        # a = combined[:i+1][1]
        # b = combined[i+1:][1]

        a = Y[:i+1]
        b = Y[i+1:]

        # print(a, " and ",b)
        # weight_a = sum(a)/(sum(a) + sum(b))
        # weight_b = 1 - weight_a

        # a = np.var(a)*weight_a
        # b = np.var(b)*weight_b

        a = np.var(a)
        b = np.var(b)

        if (a+b) < min_loss:
            idx = i
            min_loss = (a+b)

    # return [min_loss, (combined[idx][0] + combined[idx+1][0])/2]
    return [min_loss, (attr[idx] + attr[idx+1])/2]


# x = [1, 2, 4, 5]
# y = [1, 2, 4, 5]
# print(red_in_var_RI(x, y))



def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    Y = np.array(Y)
    _ , count = np.unique(Y, return_counts=True)
    prob = count/sum(count)
    entropy = 0
    for p in prob:
        entropy += p * math.log2(p)

    return -entropy

    pass


def gini_index(Y: pd.Series, attr) -> float:
    """
    Function to calculate the gini index
    """

    d={}
    for i in range(len(attr)):
        if attr[i] not in d:
            d[attr[i]] = [Y[i]]
        else:
            d[attr[i]].append(Y[i])
    ans = 0
    for i in d.values():
        _ , count = np.unique(np.array(i), return_counts=True)
        prob = count/sum(count)
        a = np.square(prob)
        ans += (sum(count)/len(Y)) * (1-sum(a))

    return ans

    pass


def gini_index_RI(Y, attr):

    # Real Input and Discrete Output

    Y = np.array(Y)
    attr = np.array(attr)
    combined = np.transpose((attr, Y))
    X = combined[combined[:, 0].argsort()]

    unique, count = np.unique(Y, return_counts=True)

    # List of 0s with length equal to number of classes in Y
    a = np.array([0 for x in range(len(unique))])
  
    # List of counts for each class; len(a) = len(b)
    b = count

    gini = 1e10

    # List of classes
    unique = list(unique)

    idx = 0
    for i in range(len(attr)-1):

        # a contains counts in upper split and b in lower split

        a[unique.index(X[i][1])] += 1
        b[unique.index(X[i][1])] -= 1

        weight_a = sum(a)/(sum(a) + sum(b))
        weight_b = 1 - weight_a

        prob =[]
        for k in a:
            prob.append(k/sum(a))

        t = 1-sum(np.square(np.array(prob)))    # Gini for upper split
        t *= weight_a   # Weighted Gini

        prob2 =[]
        for k in b:
            prob.append(k/sum(b))
        p = 1-sum(np.square(np.array(prob2)))   # Lower split
        p = t + p*weight_b  # Weighted sum of gini
        
        # Minimize Gini
        if gini > p:
            idx = i
            gini = p

    return [gini, (combined[idx][0] + combined[idx+1][0])/2]


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """

    initial_entropy = entropy(Y)
    value_entropy = 0
    unique, count = np.unique(attr, return_counts=True)
    probability = count/sum(count)
    list_attr = np.array(attr)
    Y = np.array(Y)
    for i in range(len(unique)):
        l=[]
        for j in range(len(list_attr)):
            if unique[i] == list_attr[j]:
                l.append(Y[j])
        value_entropy += probability[i]*entropy(l)
    return initial_entropy - value_entropy

    pass

def information_gain_RI(Y, attr):

    # returns (gain, split)

    if(len(Y) == 1):
        return [0, Y[0]]

    Y_new = np.array(Y)
    attr = np.array(attr)
    combined = np.transpose((attr, Y_new))
    X = combined[combined[:, 0].argsort()]  # sort based on first column of real input

    unique, count = np.unique(Y_new, return_counts=True)
    unique = list(unique)
    a = [0 for x in range(len(unique))]
    b = count
    minentropy = 1e10
    idx = 0
    for i in range(len(attr)-1):
        a[unique.index(X[i][1])] += 1
        b[unique.index(X[i][1])] -= 1

        weight_a = sum(a)/(sum(a) + sum(b))
        weight_b = 1 - weight_a
        prob = np.array(a)/sum(a)
        ent = 0
        for p in prob:
            if p > 0:
                ent += p * math.log2(p) * weight_a

        prob = np.array(b)/sum(b)
        for p in prob:
            if p > 0:
                ent += p * math.log2(p) * weight_b

        if minentropy>ent:
            idx = i
            minentropy=ent

    return [entropy(Y) - minentropy, (combined[idx][0] + combined[idx+1][0])/2]
    pass
