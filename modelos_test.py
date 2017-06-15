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

train_merge = pd.get_dummies(train_merge)

plt.hist(train['rating'], bins=10)


#ridge regression

from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[0.1, 1.0,2.0,5.0,7.0,10.0])

#partir en train y test
from sklearn.model_selection import train_test_split

indice = train.index
indice_train = 0:(round(indice.max()*70/100,0))

train_merge = train_merge.drop('directorName',axis=1)

train_df, test_df = train[]

train_df, test_df = train_test_split(train_merge, test_size = 0.1)

y = train_df.rating
x = train_df.drop('rating',axis = 1)


y_test = test_df.rating
x_test = test_df.drop('rating',axis = 1)


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
from sklearn.multioutput import MultiOutputRegressor

max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                          random_state=0))
regr_multirf.fit(x, y)

regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(x, y)

# Predict on new data
y_multirf = regr_multirf.predict(x)
y_rf = regr_rf.predict(X_test)

# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_test[:, 0], y_test[:, 1],
            c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(y_multirf[:, 0], y_multirf[:, 1],
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1],
            c="c", s=s, marker="^", alpha=a,
            label="RF score=%.2f" % regr_rf.score(X_test, y_test))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()
