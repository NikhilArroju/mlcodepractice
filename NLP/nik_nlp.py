# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:10:52 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting =3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    #review = re.sub('not ', 'no',review)
    #review = re.sub('no ', 'no',review)
    review = review.lower()
    review = review.split()
    ps =  PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]    
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=42)

from sklearn.metrics import confusion_matrix

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

#NaiveBayes
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_NB.predict(X_test)

# Making the Confusion Matrix
cm_NB = confusion_matrix(y_test, y_pred)
print("For Naive Bayes")
fun_performance(cm_NB)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy',random_state=42)
classifier_dt.fit(X_train,y_train)

y_pred = classifier_dt.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test,y_pred)
print("For CART")
fun_performance(cm_dt)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier_ranforest = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',random_state=42)
classifier_ranforest.fit(X_train,y_train)

y_pred = classifier_ranforest.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_ranforest = confusion_matrix(y_test,y_pred)
print("For Random Forests")
fun_performance(cm_ranforest)
