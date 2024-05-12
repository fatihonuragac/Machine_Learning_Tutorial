# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:47:22 2024

@author: fatihonuragac
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


df=pd.read_csv("column_2C_weka.csv")
x=df.drop("klas",axis=1)
df.klas= [1 if each == "Normal" else 0 for each in df.klas]
y_data=df.klas.values
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
x_data = mms.fit_transform(x_data)
from  sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
print("{} score={}".format(3,knn.score(x_test,y_test )))
score_list=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15),score_list)
plt.show()
print(score_list.index(max(score_list))+1)



