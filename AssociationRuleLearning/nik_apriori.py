# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:20:51 2018

@author: Nik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(20)])

for i in transactions:
    while 'nan' in i:
        i.remove('nan')
        
from apyori import apriori
#support = atleast 3 times a day for a week -->3*7/7500(total no of transactions)
rules = apriori(transactions,min_support=0.0028,min_confidence=0.2,min_lift=3,min_length=2)

results = list(rules)
for i in range(len(results)):
    print(results[i].items)
print(len(results))