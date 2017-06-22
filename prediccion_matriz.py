#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:39:23 2017

@author: pablotempone
"""

import numpy as np
import pandas as pd

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



train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#calcular distancias
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(R, metric='cosine')
item_similarity = pairwise_distances(R.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(R, item_similarity, type='item')
user_prediction = predict(R, user_similarity, type='user')

#evaluacion

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

    prediction = prediction[ground_truth.nonzero()].flatten()


print( 'User-based CF RMSE: ' + str(rmse(user_prediction, R_test)))
print( 'Item-based CF RMSE: ' + str(rmse(item_prediction, R_test)))

predict_test = predict(test_matrix,item_similarity,type='item')

#model based collaborative filtering

sparsity=round(1.0-len(train_data)/float(n_users*n_items),3)
print( 'The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

#SINGLE VALUE DESCOMPOSITION

import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(R, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print( 'User-based CF MSE: ' + str(rmse(X_pred, test_matrix)))