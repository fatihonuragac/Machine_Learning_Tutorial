# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:19:25 2024

@author: fatihonuragac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("data.csv")
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)
df.diagnosis=[1 if each=="M" else 0 for each in df.diagnosis]
x_data=df.drop(["diagnosis"],axis=1)
y=df.diagnosis.values

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
x=mms.fit_transform(x_data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
sample=x_test[1].reshape(1,-1)
print("sonuc {}".format(lr.predict(sample)))


