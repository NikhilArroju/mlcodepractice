# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:20:40 2018

@author: Nik
"""
import numpy as np
tuple1 = (1,2,3,4)
tuple2 = (5,6,7,8)

print(f'tupel 1 is {tuple1} and tuple2 is {tuple2}')

g = lambda x,y,z:list(np.arange(x,y,z))
for i in g(1,10,0.1):
    print(round(i,1))
    
class Parrot:

    # class attribute
    species = "bird"

    # instance attribute
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def speak(self):
        print("kkkccchheeggeeerrrr")

# instantiate the Parrot class
blu = Parrot("Blu", 10)
woo = Parrot("Woo", 15)

# access the class attributes
print("Blu is a {}".format(blu.species))
print("Woo is also a {}".format(woo.__class__.species))

# access the instance attributes
print("{} is {} years old".format( blu.name, blu.age))
print("{} is {} years old".format( woo.name, woo.age))

class X: pass
class Y: pass
class Z: pass

class A(X,Y): pass
class B(Y,Z): pass

class M(B,A,Z): pass

# Output:
# [<class '__main__.M'>, <class '__main__.B'>,
# <class '__main__.A'>, <class '__main__.X'>,
# <class '__main__.Y'>, <class '__main__.Z'>,
# <class 'object'>]

print(M.mro())

a= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

row_r1 = a[1,:]
row_r2 = a[1:2,:]

print(row_r1,row_r1.shape)
print(row_r2,row_r2.shape)


import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data,index = [1,2,2,4])
print (s,s[2])

# Using the previous DataFrame, we will delete a column
# using del function
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 
     'three' : pd.Series([10,20,30], index=['a','b','c'])}

df = pd.DataFrame(d)
print ("Our dataframe is:")
print (df)