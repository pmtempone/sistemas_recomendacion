#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 01:07:32 2017

@author: pablotempone
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

pd.options.display.max_columns = 10 
pd.options.display.width = 134
pd.options.display.max_rows = 20

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_train.csv",sep = ',',encoding = "ISO-8859-1")
train_user = train[['id_usuario','rating_ambiente']].groupby('id_usuario',as_index=False).count()
train_user = train_user.where(train_user.rating_ambiente>2).dropna()
train_rest = train[['id_restaurante','rating_ambiente']].groupby('id_restaurante',as_index=False).count()
train_rest = train_rest.where(train_rest.rating_ambiente>5).dropna()

train = train[train['id_usuario'].isin(train_user.id_usuario) & train['id_restaurante'].isin(train_rest.id_restaurante)]

test = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_test.csv",sep = ',',encoding = "ISO-8859-1")
matrix = pd.concat([train[['id_usuario','id_restaurante','rating_ambiente']],test[['id_usuario','id_restaurante','rating_ambiente']]]).drop_duplicates(['id_usuario','id_restaurante']).pivot('id_usuario','id_restaurante','rating_ambiente')
rest_means = matrix.mean()
user_means = matrix.mean(axis=1)
mzm = matrix-rest_means
mz = mzm.fillna(0)
del matrix
mask = -mzm.isnull()

iteration = 0
mse_last = 999



while iteration<200:
    iteration += 1
    svd = TruncatedSVD(n_components=100,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+rest_means
m = m.clip(lower=0,upper=3)

test['rating_ambiente'] = test.apply(lambda x:m[m.index==x.id_usuario][x.id_restaurante].values[0],axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.rating_ambiente.isnull())[0]
test.ix[missing,'rating_ambiente'] = user_means[test.loc[missing].id_usuario].values

#comida

matrix = pd.concat([train[['id_usuario','id_restaurante','rating_comida']],test[['id_usuario','id_restaurante','rating_comida']]]).drop_duplicates(['id_usuario','id_restaurante']).pivot('id_usuario','id_restaurante','rating_comida')
rest_means = matrix.mean()
user_means = matrix.mean(axis=1)
mzm = matrix-rest_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

iteration = 0
mse_last = 999


while iteration<200:
    iteration += 1
    svd = TruncatedSVD(n_components=100,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse


m = mz+rest_means
m = m.clip(lower=0,upper=3)

test['rating_comida'] = test.apply(lambda x:m[m.index==x.id_usuario][x.id_restaurante].values[0],axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.rating_comida.isnull())[0]
test.ix[missing,'rating_comida'] = user_means[test.loc[missing].id_usuario].values

#servicio

matrix = pd.concat([train[['id_usuario','id_restaurante','rating_servicio']],test[['id_usuario','id_restaurante','rating_servicio']]]).drop_duplicates(['id_usuario','id_restaurante']).pivot('id_usuario','id_restaurante','rating_servicio')
rest_means = matrix.mean()
user_means = matrix.mean(axis=1)
mzm = matrix-rest_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

iteration = 0
mse_last = 999


while iteration<200:
    iteration += 1
    svd = TruncatedSVD(n_components=100,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse


m = mz+rest_means
m = m.clip(lower=0,upper=3)

test['rating_servicio'] = test.apply(lambda x:m[m.index==x.id_usuario][x.id_restaurante].values[0],axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.rating_servicio.isnull())[0]
test.ix[missing,'rating_servicio'] = user_means[test.loc[missing].id_usuario].values


global_mean_ambiente = train.rating_ambiente.mean()
        
global_mean_comida = train.rating_comida.mean()

global_mean_servicio= train.rating_servicio.mean()


test['rating_ambiente'] = test.rating_ambiente.fillna(global_mean_ambiente)
test['rating_comida'] = test.rating_comida.fillna(global_mean_comida)
test['rating_servicio'] = test.rating_servicio.fillna(global_mean_servicio)



test.to_csv('pablot_15-svd_manual.csv',index=False)


