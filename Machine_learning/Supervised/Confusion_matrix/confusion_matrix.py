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
x_data = mms.fit_transform(x)
from  sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
print("{} score={}".format(3,knn.score(x_test,y_test )))


from sklearn.metrics import confusion_matrix

cf=confusion_matrix(y_test,y_predict)

import seaborn as sns
import matplotlib.pyplot as plt

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cf,annot=True,linewidths=0.5,fmt=".0f",ax=ax)
plt.show()



