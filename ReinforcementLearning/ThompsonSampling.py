# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:15:21 2018

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

#implementin Thopson Sampling
import random
ad_types_ts = 10
ads_selected_ts = []
total_reward_ts = 0
no_of_rewards_1 = [0]*ad_types_ts
no_of_rewards_0 = [0]*ad_types_ts

for n in range(len(dataset)):
    ad = 0
    max_theta = 0
    for i in range(ad_types_ts):
        beta = random.betavariate(no_of_rewards_1[i]+1,no_of_rewards_0[i]+1)
        if beta>max_theta:
            max_theta = beta
            ad = i
    ads_selected_ts.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        no_of_rewards_1[ad] = no_of_rewards_1[ad] + 1
    else:
        no_of_rewards_0[ad] = no_of_rewards_0[ad] + 1
    total_reward_ts = total_reward_ts + reward

# Visualising the results
plt.hist(ads_selected_random)
plt.title('Histogram of ads selections using Thompson Sampling')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

print("By random selection",total_reward_random)
print("Using Thompson sampling",total_reward_ts)       