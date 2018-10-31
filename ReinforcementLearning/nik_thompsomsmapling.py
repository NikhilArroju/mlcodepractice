# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:47:24 2018

@author: Nik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


no_ads = 10
no_rewards_1 = [0]*no_ads
no_rewards_0 = [0]*no_ads
total_reward = 0
for n in range(len(dataset)):
    ad = 0
    max_theta = 0
    for i in range(no_ads):
        theta = random.betavariate(no_rewards_1[i]+1,no_rewards_0[i]+1)
        if theta>max_theta:
            max_theta = theta
            ad = i
    reward = dataset.values[n,ad]
    if reward == 1:
        no_rewards_1[ad] = no_rewards_1[ad]+1
    else:
        no_rewards_0[ad] = no_rewards_0[ad]+1
    total_reward = total_reward + reward

print(total_reward)

#if random selection
total_reward_rand = 0
import random
for n in range(len(dataset)):
    ad = random.randrange(no_ads)
    reward_rand = dataset.values[n,ad]
    total_reward_rand = total_reward_rand + reward_rand
    
print(total_reward_rand)