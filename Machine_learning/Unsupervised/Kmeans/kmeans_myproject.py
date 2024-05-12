# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:10:37 2024

@author: fatihonuragac
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

# class2
x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

# class3
x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)


x=np.concatenate((x1, x2,x3),axis=0).reshape(3000, -1)
y=np.concatenate((y1, y2,y3),axis=0).reshape(3000, -1)

data=np.concatenate((x,y),axis=1)
data=pd.DataFrame(data)
data.columns=["x","y"]
#%%
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,15):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.show()
#%%
kmeans2=KMeans(n_clusters=3)
data["label"]=kmeans2.fit_predict(data)

plt.scatter(data.x,data.y)
plt.show()
















