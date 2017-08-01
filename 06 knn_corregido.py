#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:16:35 2017

@author: pablotempone
"""


#KNN
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
import pandas as pd
from surprise import GridSearch
from surprise import KNNBasic
import numpy as np

# path to dataset file

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:pass@localhost:5432/sistemas_recomendacion')

train_reducido = pd.read_sql_query('select * from ratings_train_reducido',con=engine)

####ambiente knn######

train_reducido[['id_usuario','id_restaurante','rating_ambiente','fecha']].to_csv('knn_ambiente.csv',index= False)

file_path ='knn_ambiente.csv'

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0  # no shrinkage
               }
knnbasic_ambiente = KNNBasic()

k_neig = np.array([40,45,50,60])


for i in range(0,len(k_neig)):
    knnbasic_ambiente = KNNBasic(k=k_neig[i])
    perf = evaluate(knnbasic_ambiente, data, measures=['RMSE', 'MAE'],verbose=0)
    print('K es ',k_neig[i],'media',np.array(perf['rmse']).mean())

#mejor k de ambiente es 40

knnbasic_ambiente = KNNBasic(k=40)
# Retrieve the trainset.
trainset = data.build_full_trainset()

knnbasic_ambiente.train(trainset)

from sklearn.externals import joblib
joblib.dump(knnbasic_ambiente,'knnbasic_ambiente.pkl') 


####comida knn######

train_reducido[['id_usuario','id_restaurante','rating_comida','fecha']].to_csv('knn_comida.csv',index= False)

file_path ='knn_comida.csv'

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

k_neig = np.array([5,10,15,20,40])

for i in range(0,len(k_neig)):
    knnbasic_comida = KNNBasic(k=k_neig[i])
    perf = evaluate(knnbasic_comida, data, measures=['RMSE', 'MAE'],verbose=0)
    print('K es ',k_neig[i],'media',np.array(perf['rmse']).mean())

#mejor k de ambiente es 40

knnbasic_comida = KNNBasic(k=40)
# Retrieve the trainset.
trainset = data.build_full_trainset()

knnbasic_comida.train(trainset)

from sklearn.externals import joblib
joblib.dump(knnbasic_comida,'knnbasic_comida.pkl') 

####servicio knn######

train_reducido[['id_usuario','id_restaurante','rating_servicio','fecha']].to_csv('knn_servicio.csv',index= False)

file_path ='knn_servicio.csv'

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

k_neig = np.array([40,45,50,60])

for i in range(0,len(k_neig)):
    knnbasic_servicio = KNNBasic(k=k_neig[i])
    perf = evaluate(knnbasic_servicio, data, measures=['RMSE', 'MAE'],verbose=0)
    print('K es ',k_neig[i],'media',np.array(perf['rmse']).mean())

#mejor k de ambiente es 60

knnbasic_servicio = KNNBasic(k=60)
# Retrieve the trainset.
trainset = data.build_full_trainset()

knnbasic_servicio.train(trainset)

from sklearn.externals import joblib
joblib.dump(knnbasic_servicio,'knnbasic_servicio.pkl') 


def pred_knnbasic(test):
    ##AMBIENETE
    knnbasic_ambiente = joblib.load('knnbasic_ambiente.pkl') 
    test_ambiente = pd.DataFrame()
    
    for i in range(0,len(test.index)):
        variable = pd.DataFrame(pd.Series(knnbasic_ambiente.predict(test.id_usuario.astype(str)[i],test.id_restaurante.astype(str)[i]).est).values)
        test_ambiente = test_ambiente.append(variable, ignore_index = True) 
    
    ##COMIDA
    knnbasic_comida = joblib.load('knnbasic_comida.pkl') 

    test_comida = pd.DataFrame()

    for i in range(0,len(test.index)):
        variable = pd.DataFrame(pd.Series(knnbasic_comida.predict(test.id_usuario.astype(str)[i],test.id_restaurante.astype(str)[i]).est).values)
        test_comida = test_comida.append(variable, ignore_index = True) 
        
    ##SERVICIO
    
    knnbasic_servicio = joblib.load('knnbasic_servicio.pkl') 

    test_servicio = pd.DataFrame()
    
    
    for i in range(0,len(test.index)):
        variable = pd.DataFrame(pd.Series(knnbasic_servicio.predict(test.id_usuario.astype(str)[i],test.id_restaurante.astype(str)[i]).est).values)
        test_servicio = test_servicio.append(variable, ignore_index = True) 

    test_entrega = pd.concat(objs=(test[['id_usuario','id_restaurante','fecha']],test_ambiente,test_comida,test_servicio), axis=1,ignore_index=0)
    test_entrega.columns = ['id_usuario','id_restaurante','fecha','rating_ambiente','rating_comida','rating_servicio']
    return test_entrega


entrega = pred_knnbasic(test)

entrega.to_csv('pablot-03-knnbasic.csv',index=False)