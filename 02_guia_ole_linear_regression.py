#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 23:55:24 2017

@author: pablotempone
"""

import pandas as pd
import numpy as np


train = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_train.csv",sep = ',',encoding = "ISO-8859-1")

#separar en train y test

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(train, test_size=0.03)

mean_user = train_data[['id_usuario','rating_ambiente','rating_comida','rating_servicio']].groupby('id_usuario',as_index=False).mean()
mean_user.columns = ['id_usuario','rating_ambiente_usuario','rating_comida_usuario','rating_servicio_usuario']


mean_restaurant = train_data[['id_restaurante','rating_ambiente','rating_comida','rating_servicio']].groupby('id_restaurante',as_index=False).mean()
mean_restaurant.columns = ['id_restaurante','rating_ambiente_rest','rating_comida_rest','rating_servicio_rest']

train_data = pd.merge(train_data,mean_user, how='left',left_on='id_usuario',right_on='id_usuario')
train_data = pd.merge(train_data,mean_restaurant, how='left',left_on='id_restaurante',right_on='id_restaurante')

test_data = pd.merge(test_data,mean_user, how='left',left_on='id_usuario',right_on='id_usuario')
test_data = pd.merge(test_data,mean_restaurant, how='left',left_on='id_restaurante',right_on='id_restaurante')


from sklearn import linear_model  
model = linear_model.LinearRegression()  


###regresion del ambiente
model.fit(train_data[['rating_ambiente_usuario','rating_ambiente_rest']], train_data.rating_ambiente)  

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error

test_data_completa = test_data[['id_usuario','id_restaurante','rating_ambiente_usuario','rating_ambiente_rest']].dropna()

test_data_incompleta = test_data[['id_usuario','id_restaurante','rating_ambiente_usuario','rating_ambiente_rest']][~test_data.index.isin(test_data_completa.index)]

global_mean_ambiente = train_data.rating_ambiente.mean()

test_data_incompleta['rating_ambiente_pred'] = global_mean_ambiente

y_pred = model.predict(test_data_completa[['rating_ambiente_usuario','rating_ambiente_rest']])

test_data_completa['rating_ambiente_pred'] = y_pred

test_data_pred = pd.concat([test_data_completa,test_data_incompleta]).sort_index()

print("Mean squared error: %.2f"
      % np.mean((test_data_pred.rating_ambiente_pred - test_data.rating_ambiente) ** 2))


###regresion del comida

model_comida = linear_model.LinearRegression()  

model_comida.fit(train_data[['rating_ambiente_usuario','rating_ambiente_rest']], train_data.rating_ambiente)  

# The coefficients
print('Coefficients: \n', model_comida.coef_)
# The mean squared error

test_data_completa = test_data[['id_usuario','id_restaurante','rating_ambiente_usuario','rating_ambiente_rest']].dropna()

test_data_incompleta = test_data[['id_usuario','id_restaurante','rating_ambiente_usuario','rating_ambiente_rest']][~test_data.index.isin(test_data_completa.index)]

global_mean_ambiente = train_data.rating_ambiente.mean()

test_data_incompleta['rating_ambiente_pred'] = global_mean_ambiente

y_pred = model_comida.predict(test_data_completa[['rating_ambiente_usuario','rating_ambiente_rest']])

test_data_completa['rating_ambiente_pred'] = y_pred

test_data_pred = pd.concat([test_data_completa,test_data_incompleta]).sort_index()

print("Mean squared error: %.2f"
      % np.mean((test_data_pred.rating_ambiente_pred - test_data.rating_ambiente) ** 2))

