# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:26:25 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

dataset = pd.read_csv('Churn_Modelling.csv')
dataset_homework = pd.read_csv('Churn_Modelling_HomeWorkTest.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values
X_homework = dataset_homework.iloc[:,3:13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#dummy varible
X = X[:,1:]

#Forhomeowork
X_homework[:,1] = labelencoder_X_1.transform(X_homework[:,1])
X_homework[:,2] = labelencoder_X_2.transform(X_homework[:,2])
X_homework = onehotencoder.transform(X_homework).toarray()
X_homework = X_homework[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_homework = scaler.transform(X_homework)

import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer= 'uniform', activation = 'relu', input_dim = 11))
#For dealing with overfitting
# classifier.add(Dropout(p = 0.1))

classifier.add(Dense(units = 6,kernel_initializer="uniform",activation = 'relu'))
# classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units = 1 ,kernel_initializer='uniform' ,activation = 'sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

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

y_pred_hw = classifier.predict(X_homework)
print(y_pred_hw>0.5)

#the tutorial method is
new_pred = classifier.predict(scaler.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
print(new_pred>0.5)

print("----%s seconds----"%(time.time()-start_time))
print("----%s minutes----"%((time.time()-start_time)/60))