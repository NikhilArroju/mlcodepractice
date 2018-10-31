# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:32:07 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_1 = LabelEncoder()
X[:,1] = labelencoder_1.fit_transform(X[:,1])
labelencoder_2 = LabelEncoder()
X[:,2] = labelencoder_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

#dummy variabel
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 6,activation = 'relu',kernel_initializer='uniform',input_dim = 11))
classifier.add(Dense(units = 6,activation = 'relu',kernel_initializer='uniform'))
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

def fun_performance(cm):
    TP,FP,FN,TN = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1score = 2*precision*recall/(precision+recall)
    print("Accuracy",accuracy)
    print("Recall",recall)
    print("Precison",precision)
    print("F1Score",f1score)
    
fun_performance(cm)