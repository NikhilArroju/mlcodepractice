# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:41:54 2018

@author: Nik
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 18:45:50 2018

@author: Nik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#This is my datapreprocessing template
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X,y)
y_pred = regressor.predict(6.5)





