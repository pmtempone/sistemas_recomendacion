#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 20:38:17 2017

@author: pablotempone
"""

from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
import pandas as pd
from surprise import GridSearch


# path to dataset file
file_path ='surprise_format.csv'

train = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_train.csv",sep = ',',encoding = "ISO-8859-1")
train[['id_usuario','id_restaurante','rating_ambiente','fecha']].to_csv('surprise_format.csv',index= False)


test = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/Copia de ratings_test.csv",sep = ',')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)

#grid search

param_grid = {'n_epochs': [5, 10,50,100,200], 'lr_all': [0.002, 0.005,0.007,0.01],
              'reg_all': [0.2,0.3,0.4, 0.6]}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])

grid_search.evaluate(data)

print(grid_search.best_score['RMSE'])

print(grid_search.best_params['RMSE'])

results_df = pd.DataFrame.from_dict(grid_search.cv_results)
print(results_df)

results_df.to_csv('svd_grid_search.csv')

#entrenar con todo y los mejores parametros
algo = SVD(n_epochs = 100,lr_all=0.002,reg_all=0.2)
trainset= data.build_full_trainset()
algo.train(trainset)

#predict con test

test_ambiente = pd.DataFrame()



for i in range(0,len(test.index)):
    variable = pd.DataFrame(pd.Series(algo.predict(test.id_usuario.astype(str)[i],test.id_restaurante.astype(str)[i]).est).values)
    test_ambiente = test_ambiente.append(variable, ignore_index = True)

## busqueda de rating_comida

train[['id_usuario','id_restaurante','rating_comida','fecha']].to_csv('surprise_comida.csv',index= False)

file_path ='surprise_comida.csv'

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=40)

# We'll use the famous SVD algorithm.
algo_comida = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo_comida, data, measures=['RMSE', 'MAE'])

print_perf(perf)

#grid search

param_grid = {'n_epochs': [5, 10,50,100], 'lr_all': [0.002, 0.005],
              'reg_all': [0.2,0.3,0.4]}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])

grid_search.evaluate(data)

print(grid_search.best_score['RMSE'])

print(grid_search.best_params['RMSE'])

results_df = pd.DataFrame.from_dict(grid_search.cv_results)
print(results_df)

results_df.to_csv('svd_grid_search_comida.csv')

### busqueda de rating_servicio

train[['id_usuario','id_restaurante','rating_servicio','fecha']].to_csv('surprise_servicio.csv',index= False)

file_path ='surprise_servicio.csv'

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

# We'll use the famous SVD algorithm.
algo_comida = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo_comida, data, measures=['RMSE', 'MAE'])

print_perf(perf)

#grid search

param_grid = {'n_epochs': [5, 10,50,100], 'lr_all': [0.002, 0.005],
              'reg_all': [0.2,0.3]}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])

grid_search.evaluate(data)

print(grid_search.best_score['RMSE'])

print(grid_search.best_params['RMSE'])

results_df = pd.DataFrame.from_dict(grid_search.cv_results)
print(results_df)

results_df.to_csv('svd_grid_search_ambiente.csv')

#KNN
from surprise import KNNBasic

sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0  # no shrinkage
               }
algo = KNNBasic(sim_options=sim_options)

perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)

file_path_test ='/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_test.csv'

reader = Reader(line_format='user item date_day date_month date_year date_hour date_minute date_second rating', sep=',',skip_lines=1)

test = Dataset.load_from_file(file_path_test, reader=reader)

algo.predict(test.userID[2], test.movieID[2],4)

trainset.n_users
trainset.n_items
trainset.global_mean
trainset.ur


variable = algo.predict(train.userID.astype(str)[1],train.movieID.astype(str)[1])

variable.est