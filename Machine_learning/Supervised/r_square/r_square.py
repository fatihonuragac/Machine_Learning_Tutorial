# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:34:43 2024

@author: fatihonuragac
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:09:38 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random+forest+regression+dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)  #n_estimators ağaç sayısı random_state random aldığın sayı boyutu ama sonraki aynı olsun 
rf.fit(x,y)
y_head=rf.predict(x)
from sklearn.metrics import r2_score
r2_score(y,y_head)        #r2_score bize doğruluk oranı verir 1 e ne kadar yakınsa o kadar doğru







