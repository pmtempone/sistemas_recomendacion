#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:25:56 2017

@author: pablotempone
"""

#factorizar matrices

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import pandas as pd

#fetch data and format it

train = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_train.csv",sep = ',')

#partir en train y test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(train, test_size = 0.1)


ratings = np.array(train_df.rating)
users = np.array(train_df.userID)
movies = np.array(movie_genres)

ratings_df = pd.DataFrame(train_df, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

R_df = train_df.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)
R_df.head()


R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

#singular value descomposition
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)

sigma = np.diag(sigma)

#predicciones

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)


preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
preds_df.head()

test_df