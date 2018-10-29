# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:20:33 2018

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
y = dataset.iloc[:,-2:-1].values


#feature scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])
y_pred = scaler_y.inverse_transform(y_pred)




