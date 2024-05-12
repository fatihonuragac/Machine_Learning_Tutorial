# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:22:06 2024

@author: fatihonuragac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  
df=pd.read_csv("mls-salaries-2017.csv")
x=df.base_salary.values.reshape(-1,1)
y=df.guaranteed_compensation.values.reshape(-1,1)
lin_reg=LinearRegression()
lin_reg.fit(x, y)
array=np.array([1000,2000,3000,4000,5000,6000,7000,8000,9000]).reshape(-1, 1)
y_head=lin_reg.predict(array)
plt.xlabel("base_salary")
plt.ylabel("guaranteed_compensation")
plt.plot(array,y_head,color="blue")
