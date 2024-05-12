# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:54:24 2024

@author: fatihonuragac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("linear_regression_dataset.csv",sep=";") #Dataset okuma yapılıyor

from sklearn.linear_model import LinearRegression     #sklearn kütüphanesinden linearregression fonksiyonu çağrıldı
plt.scatter(df.deneyim,df.maas,color="red")        #scatter şeklinde grafik çizildi dataset ile
plt.xlabel("deneyim")
plt.ylabel("maas")

linear_reg=LinearRegression()                       #linear regression oluşturuldu içini değer yerleştirmesi bekleniyor
x=df.deneyim.values.reshape(-1,1)       #deneyim değerleri.values fonskiyonu ile array yapıldı ama array(14,)şeklinde oluşur bunu (14,1) yapmak için reshape kullanılır -1 kullanılırsa uygun şekilde -1 olan yeri doldur demek
y=df.maas.values.reshape(-1,1)
linear_reg.fit(x,y)             #değerler fit edildi yani yerleştirildi ve artık linear çizgi hazır
b0=linear_reg.predict([[100]])#predict fonksiyonu ile tahmin edilmesi istenen derğer yazılır 
b0_=linear_reg.intercept_       #intercept ise bize b0 yani y ekseni kestiği noktayı verir
b1=linear_reg.coef_         #coef iste eğim verir
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #deneyim yıllarını array ile yazılıp gönderilirse bize sonuçları verir tahmini
y_head=linear_reg.predict(array)
plt.plot(array,y_head,color="green")



