# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:29:45 2019

@author: sirro
"""
import numpy as np

def cross_entropy(y, p_hat):
    return -(1/len(y))*np.sum(y*np.log(p_hat) + (1 - y)*np.log(1 - p_hat))