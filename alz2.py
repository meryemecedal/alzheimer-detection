# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:32:35 2020

@author: Meryem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yükleme
veriler = pd.read_csv('oasis_longitudinal.csv')
print(veriler)



#veriyi düzenleme (eksik veriler çıkartıldı)
veriler.drop('Hand',axis=1, inplace=True)
veriler.drop('Subject ID',axis=1, inplace=True)
veriler.drop('MRI ID',axis=1, inplace=True)
veriler.drop('SES',axis=1, inplace=True)
veriler.drop('MMSE',axis=1, inplace=True)
veriler.drop('Visit',axis=1, inplace=True)
print(veriler)



#cinsiyet kolonu için kategorik-numerik dönüşümü yapıldı ve veri setine eklendi
cinsiyet = veriler.iloc[:,3:4].values
print(cinsiyet)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
cinsiyet[:,0]=le.fit_transform(cinsiyet[:,0])
print(cinsiyet) #F-0;M-1

veriler['M/F'] = cinsiyet



#hastalık durumu ile ilgili kolon kategorikten numeriğe dönüştürüldü, veri setinden ayrıldı
statu = veriler.iloc[:,0:1].values
print(statu)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
statu[:,0] = le.fit_transform(veriler.iloc[:,0])
print(statu) #nondemented-2, demented-1, converted-0

veriler.drop('Group',axis=1, inplace=True)



# veriler test ve train olmak üzere ayrıldı
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(veriler, statu, test_size=0.33, random_state=0)



#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)



#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

tahmin = lr.predict(X_test)



#multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

tahmin2 = regressor.predict(X_test)



#polinom regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 1)
x_poly = poly.fit_transform(X_train)
print(x_poly)
lr2 = LinearRegression()
lr2.fit(x_poly, Y_train)

tahmin3 = lr2.predict(poly.fit_transform(X_test))



#support vector regression
from sklearn.svm import SVR

svr = SVR(kernel = 'poly')
svr.fit(X_train, Y_train)

tahmin4 = svr.predict(X_test)

"""
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, svr.predict(X_train), color = 'blue')

plt.show()
"""


#decision tree ile tahmin
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 1)
r_dt.fit(X_train, Y_train)

tahmin5 = r_dt.predict(X_test)

"""
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, r_dt.predict(X_train), color = 'blue')

plt.show()
"""



#random forest ile tahmin
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 50, random_state = 0)
rf_reg.fit(X_train, Y_train)

tahmin6 = rf_reg.predict(X_test)

"""
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, rf_reg.predict(X_train), color = 'blue')

plt.plot()
"""

from sklearn.metrics import r2_score
r2_score(Y_test, lr.predict(X_test))
r2_score(Y_test, regressor.predict(X_test))
r2_score(Y_test, lr2.predict(poly.fit_transform(X_test)))
r2_score(Y_test, svr.predict(X_test))
r2_score(Y_test, r_dt.predict(X_test))
r2_score(Y_test, rf_reg.predict(X_test))

print('Linear Regression R2 Degeri')
print(r2_score(Y_test, lr.predict(X_test)))

print('Multiple Linear Regression R2 Degeri')
print(r2_score(Y_test, regressor.predict(X_test)))

print('Polynomial Regression R2 Degeri')
print(r2_score(Y_test, lr2.predict(poly.fit_transform(X_test))))

print('SVR R2 Degeri')
print(r2_score(Y_test, svr.predict(X_test)))

print('Decision Tree R2 Degeri')
print(r2_score(Y_test, r_dt.predict(X_test)))

print('Random Forest R2 Degeri')
print(r2_score(Y_test, rf_reg.predict(X_test)))




















