# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:11:20 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#This is my datapreprocessing template
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10,random_state = 42)
regressor.fit(X,y)
y_pred = regressor.predict(6.5)