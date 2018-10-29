# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:38:08 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#dummy variable
X = X[:,1:]

import statsmodels.formula.api as sm
X = np.append(arr =  np.ones((50,1)).astype(int),values = X,axis =1)
X_opt = X[:,[0,1,2,3,4,5]]

def backward_elimination(x,sl):
    for i in range(len(x[0])):
        regressor_OLS = sm.OLS(endog = y ,exog = x).fit()
        p_values = regressor_OLS.pvalues
        max_p_value = max(p_values)
        index_maxp = list(p_values).index(max_p_value)
        if max_p_value>sl:
            x = np.delete(x,index_maxp,1)
        else:
            return x
        
X_modeled = backward_elimination(X_opt,0.05)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_modeled,y,test_size = 0.2,random_state = 42)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

print(y_pred[0:2])
