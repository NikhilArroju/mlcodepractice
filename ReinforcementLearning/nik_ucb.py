# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:08:09 2018

@author: Nik
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

no_ads = 10
no_of_sel = [0]*no_ads
total_reward = 0
sum_rewards = [0]*no_ads
for n in range(len(dataset)):
    ad = 0
    max_ucb = 0
    for i in range(no_ads):
        if no_of_sel[i]>0:
            avg_reward = sum_rewards[i]/no_of_sel[i]
            delta = math.sqrt(1.5*math.log(n+1)/no_of_sel[i])
            ucb = avg_reward+delta
        else:
            ucb = 1e400
        if ucb>max_ucb:
            max_ucb = ucb
            ad = i
    no_of_sel[ad] = no_of_sel[ad]+1
    reward = dataset.values[n,ad]
    sum_rewards[ad] = sum_rewards[ad]+reward
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
    
    