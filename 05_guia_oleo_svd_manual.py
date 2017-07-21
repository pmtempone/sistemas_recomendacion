#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:24:46 2017

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


from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(train, test_size=0.03)


#test = pd.read_csv('test_v2.csv')
matrix = pd.concat([train_data,test_data]).pivot('id_usuario','id_restaurante','rating_ambiente')

matrix = train[['id_usuario','id_restaurante','rating_ambiente']].drop_duplicates(subset=['id_usuario','id_restaurante'],keep='last').pivot('id_usuario','id_restaurante','rating_ambiente')

resto_means = matrix.mean()
user_means = matrix.mean(axis=1)
mzm = matrix-resto_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

iteration = 0
mse_last = 999
while iteration<200:
    iteration += 1
    svd = TruncatedSVD(n_components=15,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+resto_means
m = m.clip(lower=1,upper=5)

test = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_test.csv",sep = ',',encoding = "ISO-8859-1")

test['rating_ambiente_pred'] = test.apply(lambda x:m[m.index==x.id_usuario][x.id_restaurante].values[0],axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.rating.isnull())[0]
test.ix[missing,'rating'] = user_means[test.loc[missing].userID].values

test[['userID','movieID','rating']].to_csv('pablot-02-svd.csv',index=False)
