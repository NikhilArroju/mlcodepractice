# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:21:38 2018

@author: Nik
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

for i in range(len(transactions)):
    while 'nan' in transactions[i]:
        transactions[i].remove('nan')

from apyori import apriori
#support = atleast 3 times a day for a week -->3*7/7500(total no of transactions)
rules = apriori(transactions,min_support=0.0028,min_confidence=0.2,min_lift=3,min_length=2)

results = list(rules)
for i in range(len(results)):
    print(results[i].items)
print(len(results))