# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:50:56 2019

@author: sirro
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
from metrics import accuracy
from LogisticRegression import LogisticRegression
from os import path
import seaborn as sb

sb.set()


def main():
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "..", "donut.csv"))
    data = pd.read_csv(filepath, header = None)
    df = data.iloc[:,0].str.split(expand=True)
    
    #Convert data str to float
    df[0] = df[0].astype(float)
    df[1] = df[1].astype(float)
    df[2] = df[2].astype(float).astype(int)
    
    #create third dimension
    df = pd.concat([df, np.sqrt(df.iloc[:,0]**2 + df.iloc[:,1]**2)], axis=1)
    df.columns = [0, 1, 3, 2]
    
    #Divide data into variables and target
    X = df.drop([3], axis=1).to_numpy()
    y = df[3].to_numpy()
    
    color= ['red' if l == 0 else 'green' for l in y]
    
    #Plot the variables and target
    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=color, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Donut 2D Plot')
    plt.show()
    
    plt.figure()
    plt.axes(projection='3d')
    plt.scatter(X[:,0], X[:,1], X[:,2], c=color, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Donut 3D Plot')
    plt.show()
    
    log_reg = LogisticRegression()
    log_reg.fit(X, y, eta = 1, show_curve = True)
    y_hat = log_reg.predict(X)
    
    print(f"Training Accuracy: {accuracy(y, y_hat):0.4f}")
    
    x1 = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 1000)
    x2 = np.sqrt((-log_reg.b/log_reg.w[2])**2 - x1**2)
    x3 = -(np.sqrt((-log_reg.b/log_reg.w[2])**2 - x1**2))
    
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=color, alpha = 0.5)
    plt.plot(x1, x2, color = "#000000", linewidth = 2)
    plt.plot(x1, x3, color = "#000000", linewidth = 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Donut 2D Plot')
    plt.show()
    
       
    
if __name__ == "__main__":
    main()