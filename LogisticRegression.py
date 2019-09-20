# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:11:50 2019

@author: Sean Grant
"""
from activations import sigmoid
from objectives import cross_entropy
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression():
    
    def __init__(self, thresh = 0.5):
        self.thresh = thresh
    
    def fit(self, X, y, eta = 1e-3, epochs = 1e3, show_curve = False):
        epochs = int(epochs)
        N, D = X.shape
        
        self.w = np.random.randn(D)
        self.b = np.random.randn(1)
        
        J = np.zeros(epochs)
        for epoch in range(epochs):
            p_hat = self.__forward__(X)

            J[epoch] = cross_entropy(y, p_hat)
            self.w -= eta*(1/N)*X.T@(p_hat - y)
            self.b -= eta*(1/N)*np.sum(p_hat - y)
        
        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("$\mathcal{J}$")
            plt.title("Training Curve")
            plt.show()
            
    def __forward__(self, X):
        return sigmoid(X@self.w + self.b)
            
    def predict(self, X):
        return (self.__forward__(X) >= self.thresh).astype(np.int32)

if __name__ == "__main__":
    main()