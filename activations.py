# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:30:15 2019

@author: sirro
"""
import numpy as np

def sigmoid(h):
    return 1/(1 + np.exp(-h))