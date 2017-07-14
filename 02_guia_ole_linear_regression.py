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

model_comida.fit(train_data[['rating_comida_usuario','rating_comida_rest']], train_data.rating_ambiente)  

# The coefficients
print('Coefficients: \n', model_comida.coef_)
# The mean squared error

test_data_completa = test_data[['id_usuario','id_restaurante','rating_comida_usuario','rating_comida_rest']].dropna()

test_data_incompleta = test_data[['id_usuario','id_restaurante','rating_comida_usuario','rating_comida_rest']][~test_data.index.isin(test_data_completa.index)]

global_mean_comida = train_data.rating_comida.mean()

test_data_incompleta['rating_comida_pred'] = global_mean_comida

y_pred = model_comida.predict(test_data_completa[['rating_comida_usuario','rating_comida_rest']])

test_data_completa['rating_comida_pred'] = y_pred

test_data_pred = pd.concat([test_data_completa,test_data_incompleta]).sort_index()

print("Mean squared error: %.2f"
      % np.mean((test_data_pred.rating_comida_pred - test_data.rating_comida) ** 2))

###regresion del servicio

model_servicio = linear_model.LinearRegression()  

model_servicio.fit(train_data[['rating_servicio_usuario','rating_servicio_rest']], train_data.rating_ambiente)  

# The coefficients
print('Coefficients: \n', model_servicio.coef_)
# The mean squared error

test_data_completa = test_data[['id_usuario','id_restaurante','rating_servicio_usuario','rating_servicio_rest']].dropna()

test_data_incompleta = test_data[['id_usuario','id_restaurante','rating_servicio_usuario','rating_servicio_rest']][~test_data.index.isin(test_data_completa.index)]

global_mean_servicio = train_data.rating_servicio.mean()

test_data_incompleta['rating_servicio_pred'] = global_mean_servicio

y_pred = model_servicio.predict(test_data_completa[['rating_servicio_usuario','rating_servicio_rest']])

test_data_completa['rating_servicio_pred'] = y_pred

test_data_pred = pd.concat([test_data_completa,test_data_incompleta]).sort_index()

print("Mean squared error: %.2f"
      % np.mean((test_data_pred.rating_servicio_pred - test_data.rating_servicio) ** 2))


## predict entrega 1

test = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_test.csv",sep = ',',encoding = "ISO-8859-1")


def pred_reg_lineal (set_train,set_test):
    mean_user = set_train[['id_usuario','rating_ambiente','rating_comida','rating_servicio']].groupby('id_usuario',as_index=False).mean()
    mean_user.columns = ['id_usuario','rating_ambiente_usuario','rating_comida_usuario','rating_servicio_usuario']
    
    
    mean_restaurant = set_train[['id_restaurante','rating_ambiente','rating_comida','rating_servicio']].groupby('id_restaurante',as_index=False).mean()
    mean_restaurant.columns = ['id_restaurante','rating_ambiente_rest','rating_comida_rest','rating_servicio_rest']
    
    test_entrega = pd.merge(set_test,mean_user, how='left',left_on='id_usuario',right_on='id_usuario')
    test_entrega = pd.merge(test_entrega,mean_restaurant, how='left',left_on='id_restaurante',right_on='id_restaurante')
    
    train_entrega = pd.merge(set_train,mean_user, how='left',left_on='id_usuario',right_on='id_usuario')
    train_entrega = pd.merge(train_entrega,mean_restaurant, how='left',left_on='id_restaurante',right_on='id_restaurante')
    
    model_ambiente = linear_model.LinearRegression()  
    
    model_comida = linear_model.LinearRegression()  

    model_servicio = linear_model.LinearRegression()  

    model_ambiente.fit(train_entrega[['rating_ambiente_usuario','rating_ambiente_rest']], train_entrega.rating_ambiente)  
    model_comida.fit(train_entrega[['rating_comida_usuario','rating_comida_rest']], train_entrega.rating_comida)
    model_servicio.fit(train_entrega[['rating_servicio_usuario','rating_servicio_rest']], train_entrega.rating_servicio)
    
    test_data_completa = test_entrega[['id_usuario','id_restaurante','fecha','rating_ambiente_usuario','rating_ambiente_rest','rating_comida_usuario','rating_comida_rest','rating_servicio_usuario','rating_servicio_rest']].dropna()
    
    test_data_incompleta = test_entrega[['id_usuario','id_restaurante','fecha','rating_ambiente_usuario','rating_ambiente_rest','rating_comida_usuario','rating_comida_rest','rating_servicio_usuario','rating_servicio_rest']][~test_entrega.index.isin(test_data_completa.index)]
        
    global_mean_ambiente = set_train.rating_ambiente.mean()
        
    global_mean_comida = set_train.rating_comida.mean()

    global_mean_servicio= set_train.rating_servicio.mean()

    test_data_incompleta['rating_ambiente_pred'] = round(global_mean_ambiente,2)
    test_data_incompleta['rating_comida_pred'] = round(global_mean_comida,2)
    test_data_incompleta['rating_servicio_pred'] = round(global_mean_servicio,2)
    
        
    pred_ambiente =  np.round(model_ambiente.predict(test_data_completa[['rating_ambiente_usuario','rating_ambiente_rest']]))
    
    pred_comida =  np.round(model_comida.predict(test_data_completa[['rating_comida_usuario','rating_comida_rest']]))
    
    pred_servicio =  np.round(model_servicio.predict(test_data_completa[['rating_servicio_usuario','rating_servicio_rest']]))

    test_data_completa['rating_ambiente_pred'] = np.round(pred_ambiente)
    test_data_completa['rating_comida_pred'] = np.round(pred_comida)
    test_data_completa['rating_servicio_pred'] = np.round(pred_servicio)
    
    test_data_pred = pd.concat([test_data_completa,test_data_incompleta]).sort_index()
    
    test_data_pred['rating_ambiente_pred'] = np.absolute(np.where(test_data_pred['rating_ambiente_pred']<0,0,np.where(test_data_pred['rating_ambiente_pred']>3,3,test_data_pred['rating_ambiente_pred'])))
    test_data_pred['rating_comida_pred'] = np.absolute(np.where(test_data_pred['rating_comida_pred']<0,0,np.where(test_data_pred['rating_comida_pred']>3,3,test_data_pred['rating_comida_pred'])))
    test_data_pred['rating_servicio_pred'] = np.absolute(np.where(test_data_pred['rating_servicio_pred']<0,0,np.where(test_data_pred['rating_servicio_pred']>3,3,test_data_pred['rating_servicio_pred'])))

    test_data_pred.columns = ['id_usuario', 'id_restaurante', 'fecha', 'rating_ambiente_usuario',
       'rating_ambiente_rest', 'rating_comida_usuario', 'rating_comida_rest',
       'rating_servicio_usuario', 'rating_servicio_rest',
       'rating_ambiente', 'rating_comida', 'rating_servicio']

    
    return test_data_pred


entrega = pred_reg_lineal(train,test)

entrega[['id_usuario','id_restaurante','fecha','rating_ambiente', 'rating_comida', 'rating_servicio']].to_csv('pablot-01-rl.csv',index=False)