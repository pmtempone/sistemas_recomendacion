#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 22:01:48 2017

@author: pablotempone
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score


# Create a random dataset
rng = np.random.RandomState(1)

cols = ['fecha'] + list(train.loc[:,'fecha_alta':'ambiente_oleo'])

x, y = train[cols], train.rating_ambiente

from sklearn import preprocessing
from collections import defaultdict
d = defaultdict(preprocessing.LabelEncoder)

x[['precio']] = x[['precio']].fillna(0)
x[['fotos']] = x[['fotos']].fillna(0)
x[['comida_oleo']] = x[['comida_oleo']].fillna(0)
x[['servicio_oleo']] = x[['servicio_oleo']].fillna(0)
x[['ambiente_oleo']] = x[['ambiente_oleo']].fillna(0)


# Encoding the variable .fillna('0')
fit = x.apply(lambda x: d[x.name].fit_transform(x.fillna('0')))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
x.apply(lambda x: d[x.name].transform(x))

parameters = {'n_estimators':(10, 15,20,30)}
rf_ambiente = RandomForestRegressor()
clf = GridSearchCV(cv=5,error,rf_ambiente, parameters,n_jobs=-1)
clf.fit(fit, y)

sorted(clf.cv_results_.keys())

clf.cv_results_
clf.best_estimator_
clf.best_score_

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100,n_jobs=-1)
score = cross_val_score(estimator, fit, y,scoring='neg_mean_squared_error').mean()
print("Score with the entire dataset = %.2f" % score)

from sklearn.metrics import mean_squared_error

rf_ambiente = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

rf_ambiente.fit(fit,y)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(pd.concat((fit,y),axis=1), test_size = 0.05)

y = train_df.rating_ambiente
x = train_df.drop('rating_ambiente',axis = 1)



rf_ambiente = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

y_test = test_df.rating_ambiente
x_test = test_df.drop('rating_ambiente',axis = 1)


rf_ambiente.fit(x,y)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((rf_ambiente.predict(x_test) - y_test) ** 2))

mean_squared_error(rf_ambiente.predict(x_test),y_test)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rf_ambiente.score(x_test, y_test))


#rating comida



