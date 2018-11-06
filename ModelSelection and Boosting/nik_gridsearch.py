# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:34:18 2018

@author: Nik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier,X = X_train,y=y_train,cv= 10)
print(accuracies.mean(),accuracies.std())

#range test
'''
for i in range(0,10):
    print(i)
for i in np.arange(0.0,10.0,0.1):
    print(i)
''' 
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000],'kernel':['linear'],'gamma':['auto']},
               {'C':[1,10,100,1000],'kernel':['rbf','poly','sigmoid'],'gamma':list(np.arange(0.0,1.0,0.1))}]

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print(best_accuracy,best_parameters)
