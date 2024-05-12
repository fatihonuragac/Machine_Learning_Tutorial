# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:46:16 2024

@author: fatihonuragac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
df=pd.read_csv("Pokemon.csv")


df.drop(['Name','Legendary'], axis=1, inplace=True)
df.drop(df.columns[[1,2]],axis=1,inplace=True)
print(df.corr())  #bu komut columsların birbiri ile sayısal olarak bağlantı oranını verir





x=np.array(df.loc[:,["Speed","Sp. Atk"]])
y=np.array(df.loc[:,"Attack"]).reshape(-1,1)
x_=np.array(df.loc[:,["HP","Defense"]])
mlp=LinearRegression()
mlp.fit(x,y)
y_predict=mlp.predict(x_)
R2=r2_score(y, y_predict)
plt.title("R2={}".format(R2))
plt.scatter(y_predict,y)
plt.show()

