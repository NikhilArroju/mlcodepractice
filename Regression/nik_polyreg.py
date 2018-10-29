# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:16:48 2018

@author: Nik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X,y)
print(lin_regressor.predict([[6.5]]))

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_regressor2 = LinearRegression()
lin_regressor2.fit(X_poly,y)
print(lin_regressor2.predict(poly_reg.fit_transform([[6.5]])))

