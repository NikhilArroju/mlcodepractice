# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:56:14 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing the random selection
import random
ad_types_random = 10
ads_selected_random = []
total_reward_random = 0
for i in range(len(dataset)):
    ad = random.randrange(ad_types_random)
    ads_selected_random.append(ad)
    reward = dataset.values[i,ad]
    total_reward_random = total_reward_random + reward
    
# Visualising the results
plt.hist(ads_selected_random)
plt.title('Histogram of ads selections at Random')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()  
    
#Implementing Upper Conifidence Bound method
import math
ad_types_ucb = 10
ads_selected_ucb = [] 
total_reward_ucb = 0
no_of_selections = [0]*ad_types_ucb
sum_of_rewards = [0]*ad_types_ucb

for n in range(len(dataset)):
    ad = 0
    max_upper_bound = 0
    for i in range(ad_types_ucb):
        if no_of_selections[i]>0:
            avg_reward = sum_of_rewards[i]/no_of_selections[i]
            delta_i = math.sqrt(1.5*math.log(n+1)/no_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound>max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected_ucb.append(ad)
    no_of_selections[ad] = no_of_selections[ad]+1
    reward = dataset.values[n,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward_ucb = total_reward_ucb + reward

print(total_reward_ucb)

# Visualising the results
plt.hist(ads_selected_ucb)
plt.title('Histogram of ads selections by UCB')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
            
print("By random selection",total_reward_random)
print("Using UCB",total_reward_ucb)      

    