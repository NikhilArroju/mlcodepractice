# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:46:43 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
#Important use only transform for X_test 
X_test = scaler_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)

X_g = dataset.iloc[:,1:4].values
y_g = dataset.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X_g[:,0] = labelencoder.fit_transform(X_g[:,0])

from sklearn.model_selection import train_test_split
X_train_g,X_test_g,y_train_g,y_test_g = train_test_split(X_g,y_g,test_size = 0.2,random_state = 42)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler_X_g = StandardScaler()
X_train_g = scaler_X.fit_transform(X_train_g)
X_test_g = scaler_X.fit_transform(X_test_g)

from sklearn.linear_model import LogisticRegression
classifier_g = LogisticRegression()
classifier_g.fit(X_train_g,y_train_g)

y_pred_g = classifier_g.predict(X_test_g)

from sklearn.metrics import confusion_matrix
cm_g = confusion_matrix(y_test_g,y_pred_g)

print(cm_g,cm)
