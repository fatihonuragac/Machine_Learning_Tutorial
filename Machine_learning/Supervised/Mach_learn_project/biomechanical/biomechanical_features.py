# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:06:42 2024

@author: fatihonuragac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("column_2C_weka.csv")
x=df.pelvic_incidence.values.reshape(-1,1)
y=df.sacral_slope.values.reshape(-1,1)
plt.scatter(x, y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x,y)
x_= np.linspace(min(x), max(x),310).reshape(-1,1)
y_head=lr.predict(x_)

from sklearn.metrics import r2_score

print(r2_score(y,y_head))
plt.plot(x_,y_head,color="black",linewidth=1)
    