#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 00:52:11 2017

@author: pablotempone
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pandas as pd


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

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(pd.concat((fit,y),axis=1), test_size = 0.05)

y = train_df.rating_ambiente
x = train_df.drop('rating_ambiente',axis = 1)

y_test = test_df.rating_ambiente
x_test = test_df.drop('rating_ambiente',axis = 1)

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(x, y)
mse = mean_squared_error(y_test, clf.predict(x_test))
print("MSE: %.4f" % mse)


from sklearn.externals import joblib
joblib.dump(clf,'gbr_ambiente.pkl') 

#con esto se carga

svd_servicio = joblib.load('svd_servicio.pkl') 

