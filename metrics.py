from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    same = 0
    total = 0
    y_hat = np.array(y_hat)
    y = np.array(y)
    for i in y_hat:
        if i == y[total]:
            same+=1
        total+=1

    acc = same/total
    return acc
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """

    same = 0
    total = 0
    for idx,elem in y_hat.iteritems():
        if elem == cls:
            total+=1
            if elem == y[idx]:
                same+=1
    if total == 0:
        print("Class is not predicted")
        return
    prec = same/total
    return prec

    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """

    same = 0
    total = 0
    for idx,elem in y.iteritems():
        if elem == cls:
            total+=1
            if elem == y_hat[idx]:
                same+=1
    if total == 0:
        print("Class is not in ground truth")
        return
    rec = same/total
    return rec

    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    y_hat = np.array(y_hat)
    y = np.array(y)
    total_square_err = 0
    count = 0
    for i in y:
        err = y[count] - y_hat[count]
        square_err = err**2
        total_square_err += square_err
        count += 1
        
    mse = total_square_err/count
    rmse_error = mse**0.5
    return rmse_error

    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """

    total_error = 0
    count = 0
    for idx,elem in y.iteritems():
        err = elem - y[idx]
        abs_err = abs(err)
        total_error += abs_err
        count+=1

    mean_abs_err = total_error/count
    return mean_abs_err

    pass
