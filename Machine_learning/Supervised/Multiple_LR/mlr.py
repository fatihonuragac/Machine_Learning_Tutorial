# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:01:27 2024

@author: fatihonuragac
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df=multiple_linear_regression_datasetcsv
y=df.maas.values.reshape(-1,1)
x=df.iloc[:,[0,2]]
mlp=LinearRegression()
mlp.fit(x,y)
mlp.predict(np.array([[10,25],[15,30],[100,100]]))
