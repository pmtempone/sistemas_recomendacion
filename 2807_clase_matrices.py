#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:47:26 2017

@author: pablotempone
"""

import numpy as np
import pandas as pd
from scikits.crab.models import MatrixPreferenceDataModel

from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity

train = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_train.csv",sep = ',')

test = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/Copia de ratings_test.csv",sep = ',')

n_users = train.userID.unique().shape[0]
n_items = train.movieID.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(train, test_size=0.10)

#Create two user-item matrices, one for training and another for testing

R_df = train_data.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)
R_df.head()


R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)


R_df_test = test_data.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)
R_df_test.head()

R_test = R_df_test.as_matrix()

test_matrix = test.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)
test_matrix = test_matrix.as_matrix()


#calcular distancias
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(R, metric='correlation')

user_similarity.min()
item_similarity = pairwise_distances(R.T, metric='cosine')

mezcla = np.dot(user_similarity,R)
matriz = user_similarity.dot(R)


mean_user_rating = ratings.mean(axis=1)
#You use np.newaxis so that mean_user_rating has same format as ratings
ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred



train_completo = user_similarity*4.5+0.5

from sklearn.preprocessing import normalize

matriz_norm = normalize(matriz)