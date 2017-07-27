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

train_means = train[['id_usuario','rating_ambiente','rating_comida','rating_servicio']].groupby('id_usuario',as_index=False).mean()
train_means.columns = ['id_usuario','usuario_mean_ambiente','usuario_mean_comida','usuario_mean_servicio']

train_means_rest = train[['id_restaurante','rating_ambiente','rating_comida','rating_servicio']].groupby('id_restaurante',as_index=False).mean()
train_means_rest.columns = ['id_restaurante','rest_mean_ambiente','rest_mean_comida','rest_mean_servicio']



train = pd.merge(train,train_means,how = 'left',left_on = 'id_usuario',right_on = 'id_usuario')

train = pd.merge(train,train_means_rest,how = 'left',left_on = 'id_restaurante',right_on = 'id_restaurante')

train.to_sql('train_completo_v3',engine,index=False)

cols = list(train.loc[:,'id_usuario':'fecha']) + list(train.loc[:,'fecha_alta':'rest_mean_servicio'])

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



parameters = {'n_estimators':(10, 15,20,30)}
rf_ambiente = RandomForestRegressor()
clf = GridSearchCV(rf_ambiente, parameters,n_jobs=-1,cv=5)
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

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rf_ambiente.score(x_test, y_test))

#buen rmse, entrenar modelo total



y = train.rating_ambiente

rf_ambiente.fit(fit,y)
from sklearn.externals import joblib
joblib.dump(rf_ambiente,'rf_ambiente.pkl') 


#rating comida


from sklearn.model_selection import train_test_split

x, y = train[cols], train.rating_comida

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



train_df, test_df = train_test_split(pd.concat((fit,y),axis=1), test_size = 0.05)

y = train_df.rating_comida
x = train_df.drop('rating_comida',axis = 1)



rf_comida = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

y_test = test_df.rating_comida
x_test = test_df.drop('rating_comida',axis = 1)


rf_comida.fit(x,y)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((rf_comida.predict(x_test) - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rf_comida.score(x_test, y_test))

y = train.rating_comida

rf_comida.fit(fit,y)
from sklearn.externals import joblib
joblib.dump(rf_comida,'rf_comida.pkl') 

#rating servicio


from sklearn.model_selection import train_test_split

x, y = train[cols], train.rating_servicio

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



train_df, test_df = train_test_split(pd.concat((fit,y),axis=1), test_size = 0.05)

y = train_df.rating_servicio
x = train_df.drop('rating_servicio',axis = 1)



rf_servicio = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=-1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

y_test = test_df.rating_servicio
x_test = test_df.drop('rating_servicio',axis = 1)


rf_servicio.fit(x,y)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((rf_servicio.predict(x_test) - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rf_servicio.score(x_test, y_test))

y = train.rating_servicio

rf_servicio.fit(fit,y)
from sklearn.externals import joblib
joblib.dump(rf_servicio,'rf_servicio.pkl') 


#si no tengo las medias las vuelvo a generar

train_means = train[['id_usuario','rating_ambiente','rating_comida','rating_servicio']].groupby('id_usuario',as_index=False).mean()
train_means.columns = ['id_usuario','usuario_mean_ambiente','usuario_mean_comida','usuario_mean_servicio']

train_means_rest = train[['id_restaurante','rating_ambiente','rating_comida','rating_servicio']].groupby('id_restaurante',as_index=False).mean()
train_means_rest.columns = ['id_restaurante','rest_mean_ambiente','rest_mean_comida','rest_mean_servicio']



train = pd.merge(train,train_means,how = 'left',left_on = 'id_usuario',right_on = 'id_usuario')

train = pd.merge(train,train_means_rest,how = 'left',left_on = 'id_restaurante',right_on = 'id_restaurante')


## funcion de prediccion

test = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_test.csv",sep = ',',encoding = "ISO-8859-1")


def pred_rfr(test):
    
    #cargar engine en script 02
    test.to_sql('test_entrega',engine,index=False)
    
    test_entrega = pd.read_sql_query('select a.*,b.edad,b.fecha_alta,b.genero,b.tipo, c.localidad, cocina, precio, c.latitud, c.longitud, fotos, premios, "Ir en pareja", "Ir con amigos", "Comer con buenos tragos", "Llevar extranjeros", "Escuchar música", "Comer sin ser visto", "Comer al aire libre", "Comer solo", "Reunión de negocios", "Salida de amigas", "Comer bien gastando poco", "Ir con la familia", "Comer tarde", "Comer sano ", "Merendar", "Comer mucho", "Ir con chicos", "American Express", "Cabal", "Diners", "Electrón", "Maestro", "Mastercard", "Tarjeta Naranja", "Visa",telefono,comida_oleo,servicio_oleo,ambiente_oleo from test_entrega a left join usuarios b on ( a.id_usuario=b.id_usuario) left join rest_campos_v3 c  on (cast(a.id_restaurante as text)=c.id_restaurante)',engine)
    #preprocesamiento de test
    from sklearn import preprocessing
    from collections import defaultdict
    d = defaultdict(preprocessing.LabelEncoder)

    test_entrega[['precio']] = test_entrega[['precio']].fillna(0)
    test_entrega[['fotos']] = test_entrega[['fotos']].fillna(0)
    test_entrega[['comida_oleo']] = test_entrega[['comida_oleo']].fillna(0)
    test_entrega[['servicio_oleo']] = test_entrega[['servicio_oleo']].fillna(0)
    test_entrega[['ambiente_oleo']] = test_entrega[['ambiente_oleo']].fillna(0)
    
    train_means = train[['id_usuario','rating_ambiente','rating_comida','rating_servicio']].groupby('id_usuario',as_index=False).mean()
    train_means.columns = ['id_usuario','usuario_mean_ambiente','usuario_mean_comida','usuario_mean_servicio']

    train_means_rest = train[['id_restaurante','rating_ambiente','rating_comida','rating_servicio']].groupby('id_restaurante',as_index=False).mean()
    train_means_rest.columns = ['id_restaurante','rest_mean_ambiente','rest_mean_comida','rest_mean_servicio']

    

    test_entrega = pd.merge(test_entrega,train_means,how = 'left',left_on = 'id_usuario',right_on = 'id_usuario')

    test_entrega = pd.merge(test_entrega,train_means_rest,how = 'left',left_on = 'id_restaurante',right_on = 'id_restaurante')

    test_entrega[['usuario_mean_ambiente']] = test_entrega[['usuario_mean_ambiente']].fillna(0)
    test_entrega[['usuario_mean_comida']] = test_entrega[['usuario_mean_comida']].fillna(0)
    test_entrega[['usuario_mean_servicio']] = test_entrega[['usuario_mean_servicio']].fillna(0)
    
    test_entrega[['rest_mean_ambiente']] = test_entrega[['rest_mean_ambiente']].fillna(0)
    test_entrega[['rest_mean_comida']] = test_entrega[['rest_mean_comida']].fillna(0)
    test_entrega[['rest_mean_servicio']] = test_entrega[['rest_mean_servicio']].fillna(0)
    
    # Encoding the variable .fillna('0')
    test_final = test_entrega.apply(lambda x: d[x.name].fit_transform(x.fillna('0')))

    cols = list(test_final.loc[:,'id_usuario':'fecha']) + list(test_final.loc[:,'fecha_alta':'rest_mean_servicio'])

    
    rf_ambiente = joblib.load('rf_ambiente.pkl') 
    
    
    pred_ambiente =  np.round(rf_ambiente.predict(test_final[cols]))

    
    ##COMIDA
    rf_comida = joblib.load('rf_comida.pkl') 
    
    pred_comida =  np.round(rf_comida.predict(test_final[cols]))

        
    ##SERVICIO
    
    rf_servicio = joblib.load('rf_servicio.pkl') 
    
    pred_servicio =  np.round(rf_servicio.predict(test_final[cols]))

    test['rating_ambiente_pred'] = np.round(pred_ambiente)
    test['rating_comida_pred'] = np.round(pred_comida)
    test['rating_servicio_pred'] = np.round(pred_servicio)

    test_entrega = test[['id_usuario','id_restaurante','fecha','rating_ambiente_pred','rating_comida_pred','rating_servicio_pred']]
    test_entrega.columns = ['id_usuario','id_restaurante','fecha','rating_ambiente','rating_comida','rating_servicio']
    return test_entrega


entrega = pred_rfr(test)

entrega.to_csv('pablot-03-rfr.csv',index=False)
