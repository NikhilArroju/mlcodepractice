# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:26:36 2018

@author: Nik
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=42)
regressor.fit(X,y)

print(regressor.predict([[6.5]]))