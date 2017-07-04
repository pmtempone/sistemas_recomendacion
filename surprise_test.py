#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 20:15:14 2017

@author: pablotempone
"""
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
import pandas as pd

# path to dataset file
file_path ='/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_train.csv'

train = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_train.csv",sep = ',')

test = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/Copia de ratings_test.csv",sep = ',')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)

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