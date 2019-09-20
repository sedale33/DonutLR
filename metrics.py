# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:28:44 2019

@author: sirro
"""
import numpy as np

def accuracy(y, y_hat):
    return np.mean(y == y_hat)

def R2(y, y_hat):
    """R-squared"""
    return (1 - (sum((y-y_hat)**2)/(sum((y-np.mean(y))**2))))

def MAE(y, yhat):
    """ Mean Absolute Error """
    return np.mean(np.abs(y - yhat))