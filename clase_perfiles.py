#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:46:01 2017

@author: pablotempone
"""

import pandas as pd
import numpy as np

total = train.append(test)

total = pd.merge(total,movie_genres,how='left',left_on='movieID',right_on='movieID')

usuarios_generos = pd.get_dummies(total[['userID','genre']])

usuarios_generos = usuarios_generos.groupby('userID',as_index=False).sum()

usuarios_generos["sum"] = usuarios_generos.sum(axis=1)

usuarios_generos_perfil = usuarios_generos.loc[:,'genre_Action':'genre_Western'].div(usuarios_generos["sum"], axis=0)

usuarios_generos_perfil.index = usuarios_generos.userID

train_total, test_total = total[0:770088], total[770089:8555598]

train_generos = pd.merge(train,usuarios_generos,how='left',left_on='userID',right_on='userID')

test_generos = pd.merge(test,usuarios_generos,how='left',left_on='userID',right_on='userID')


from sklearn.metrics.pairwise import pairwise_distances

#matriz coseno de usuarios

user_matrix = usuarios_generos.as_matrix()


user_similarity = pairwise_distances(user_matrix, metric='cosine')

#matriz coseno de movies

movies_genres = pd.get_dummies(movie_genres)

movies_genres = movies_genres.groupby('movieID',as_index=False).sum()

movies_genres["sum"] = movies_genres.sum(axis=1)

movies_generos_perfil = movies_genres.loc[:,'genre_Action':'genre_Western'].div(movies_genres["sum"], axis=0)

movies_generos_perfil.index = movies_genres.movieID

movie_matrix = movies_generos_perfil.as_matrix()


movie_similarity = pairwise_distances(movie_matrix, metric='cosine')

#predecir en base a los puntajes

R_df = train_df.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)
R_df.head()

R_df_na = train_df.pivot(index = 'userID', columns ='movieID', values = 'rating')
R_na = R_df_na.as_matrix()

R = R_df.as_matrix()
user_ratings_mean = np.nanmean(R_na, axis = 1)

user_similarity = pairwise_distances(R, metric='cosine')



def predict(ratings,ratings_na, similarity, type='user'):

    if type == 'user':
        mean_user_rating = np.nanmean(ratings_na,axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

prueba_mean
mean_user_rating = R_na.nanmean(axis=1)

ratings_diff = (R - user_ratings_mean[:, np.newaxis])


promedios = user_ratings_mean[:, np.newaxis]

matriz = user_similarity.dot(ratings_diff)

divisor = np.array([np.abs(user_similarity).sum(axis=1)]).T

division = matriz/divisor


user_prediction = predict(R,R_na, user_similarity, type='user')


#evaluacion

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

    #prediction = prediction[ground_truth.nonzero()].flatten()


print( 'basado en similaridad CF RMSE: ' + str(rmse(user_prediction, R)))


#normalizado los valores y modificado columnas e indices

R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R, k = 50)

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)

normalized = (preds_df-min(preds_df))/(max(preds_df)-min(preds_df))

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
x_scaled = min_max_scaler.fit_transform(preds_df)
df = pd.DataFrame(x_scaled,columns=train.movieID.unique(),index=train.userID.unique())