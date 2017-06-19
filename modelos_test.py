#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 19:08:23 2017

@author: pablotempone
"""
import pandas as pd
import psycopg2
import matplotlib as mp
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn import linear_model




train = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_train.csv",sep = ',')

movie_actors = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_actors.csv",sep = ',')
movie_countries = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_countries.csv",sep = ',')
movie_directors = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_directors.csv",sep = ',')
movie_genres = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_genres.csv",sep = ',')
movie_imdb = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_imdb.csv",sep = ',')
movie_locations = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_locations.csv",sep = ',')
test = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/Copia de ratings_test.csv",sep = ',')

train_merge = pd.merge(train,movie_countries,how='left',left_on='movieID',right_on='movieID')
train_merge = pd.merge(train_merge,movie_directors,how='left',left_on='movieID',right_on='movieID')

train_merge = train_merge.drop('directorID',axis = 1)


test_merge = pd.merge(test,movie_countries,how='left',left_on='movieID',right_on='movieID')
test_merge = pd.merge(test_merge,movie_directors,how='left',left_on='movieID',right_on='movieID')

test_merge = test_merge.drop('directorID',axis = 1)

total = train_merge.append(test_merge)

total = pd.get_dummies(total)


train_merge = pd.get_dummies(train_merge)


test_merge = pd.merge(test,movie_countries,how='left',left_on='movieID',right_on='movieID')
test_merge = pd.merge(test_merge,movie_directors,how='left',left_on='movieID',right_on='movieID')

test_merge = test_merge.drop('directorID',axis = 1)

test_merge = pd.get_dummies(test_merge)

plt.hist(train['rating'], bins=10)


#ridge regression

reg = linear_model.RidgeCV(alphas=[0.1, 1.0,2.0,5.0,7.0,10.0])

#partir en train y test
from sklearn.model_selection import train_test_split

indice = train.index


train_total, test_total = total[0:770088], total[770089:8555598]

del train_merge

train_df, test_df = train_test_split(train_total, test_size = 0.1)

del train_total

y = train_df.rating
x = train_df.drop('rating',axis = 1)


y_test = test_df.rating
x_test = test_df.drop('rating',axis = 1)

del train_df,test_df
del test_merge
del train,movie_actors,test,movie_locations
del total

reg.fit(x,y)

reg.coef_

reg.alpha_ 

prediccion = round(reg.predict(x_test),1)



# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((reg.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % reg.score(x_test, y_test))

test = test.drop('rating',axis = 1)

prediccion_test = reg.predict(test)


prediccion_test.max()
prediccion_test.min()

prediccion_test = np.round_(prediccion_test, decimals=1, out=None)

#random forest reggresion

from sklearn.ensemble import RandomForestRegressor

max_depth = 30


regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(x, y)

import pickle

# save the model to disk
filename = '/Users/pablotempone/sistemas_recomendacion/sistemas_recomendacion/rf_reg_01.sav'
pickle.dump(regr_rf, open(filename, 'wb'))

# Predict on new data


y_rf = regr_rf.predict(x_test)

np.round(y_rf,1)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((np.round(y_rf,1) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr_rf.score(x_test, y_test))

test_merge_pred = test_total.drop('rating',axis = 1)

predict_test = regr_rf.predict(test_merge_pred)

predict_test = np.round(predict_test,1)

predict_test.max()
predict_test.min()

test = test.drop('rating',axis=1)

test['rating'] = predict_test

test[['userID','movieID','rating']].to_csv('pablot-01-rf.csv',index=False)