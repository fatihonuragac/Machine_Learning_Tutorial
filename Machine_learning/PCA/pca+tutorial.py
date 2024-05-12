# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:04:59 2018

@author: user
"""

from sklearn.datasets import load_iris
import pandas as pd

# %%
iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y

x = data
#%%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)


#%%
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test).reshape(-1,1)

print("accuracy: ",nb.score(x_test,y_test.reshape(-1,1)))



#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, whiten= True )  # whitten = normalize
pca.fit(x)

x_pca = pca.transform(x)

print("variance ratio: ", pca.explained_variance_ratio_)

print("sum: ",sum(pca.explained_variance_ratio_))


x1_train,x1_test,y1_train,y1_test=train_test_split(x_pca,y,test_size=0.1,random_state=42)
nb = GaussianNB()
nb.fit(x1_train,y1_train)

y_pred = nb.predict(x1_test).reshape(-1,1)

print("accuracy: ",nb.score(x1_test,y1_test.reshape(-1,1)))


#%% 2D

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
























