#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 21:27:31 2017

@author: pablotempone
"""

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import linear_model

svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models

#se separan los generos

train_generos_tipo = train_merge.genres.str.split('|', expand=True)

train_merge_svm = train_merge

train_merge_svm['genero1'] = train_generos_tipo[[0]]
train_merge_svm['genero2'] = train_generos_tipo[[1]]
train_merge_svm = train_merge_svm.drop('timestamp',axis=1)

train_merge_svm = train_merge_svm.drop('title',axis = 1)
train_merge_svm = train_merge_svm.drop('genres',axis = 1)

train_new = pd.get_dummies(train_merge_svm)
y = train_new.rating
x = train_new.drop('rating',axis = 1)

test_merge_svm = test_merge.drop('title',axis = 1)

test_merge_svm = pd.get_dummies(test_merge_svm)

svr_rbf.fit(x,y) # fitting the data points in the models
svr_lin.fit(x, y)
svr_poly.fit(dates, prices)

#preparo test

test_generos_tipo = test_merge.genres.str.split('|', expand=True)

test_merge_svm = test_merge

test_merge_svm['genero1'] = test_generos_tipo[[0]]
test_merge_svm['genero2'] = test_generos_tipo[[1]]

test_merge_svm = test_merge_svm.drop('title',axis = 1)
test_merge_svm = test_merge_svm.drop('genres',axis = 1)

test_new = pd.get_dummies(test_merge_svm)
y = train_new.rating
x = train_new.drop('rating',axis = 1)

test_merge_svm = test_merge.drop('title',axis = 1)

test_merge_svm = pd.get_dummies(test_merge_svm)

predict = svr_rbf.predict(test_new)

test_entrega_svm = test[[0,1]]
test_entrega_svm['rating'] = predict

test_entrega_svm.to_csv('entrega_pablo_02_svm.csv',index=False)

# save the model to disk
filename = '/Users/pablotempone/sistemas_recomendacion/sistemas_recomendacion/svm_rbf.sav'
pickle.dump(svr_rbf, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)



#regresion lineal
#partir en train y test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(train_new, test_size = 0.3)

y = train_df.rating
x = train_df.drop('rating',axis = 1)


reg = linear_model.LinearRegression()

y_test = test_df.rating
x_test = test_df.drop('rating',axis = 1)


reg.fit(x,y)

reg.coef_

# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((reg.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % reg.score(x_test, y_test))

# Plot outputs
plt.scatter(x_test['userId'], y_test,  color='black')
plt.plot(x_test['userId'], reg.predict(x_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()