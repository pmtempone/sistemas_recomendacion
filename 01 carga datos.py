# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn import linear_model

train = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/ratings_train.csv",sep = ',')

test = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/ratings_test.csv",sep = ',')

imdb_movie = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/imdb_movie_metadata.csv",sep = ',')

movies = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movies.csv",sep = ',')


plt.hist(train['rating'], bins=5)

list(imdb_movie)
train.head()

test_entrega_4 = test
test_entrega_4['rating'] = 4
              
test_entrega_4.to_csv('test_04.csv',index=False)

#imdb_movie['id_movie'] = re.sub('http://www.imdb.com/title/tt','',imdb_movie['movie_imdb_link'])

train_merge = pd.merge(train,movies)

test_merge = pd.merge(test,movies)

np.mean(train_merge['rating'])

valor_genero = train_merge[['rating','genres']].groupby('genres').mean()
valor_genero['genres'] = valor_genero.index
        
test_merge_2 = pd.merge(test_merge[['userId','movieId','genres']],valor_genero,how='left',left_on='genres',right_on='genres')

test_merge_2['rating'] = test_merge_2['rating'].fillna(3.87)

test_merge_2[['userId','movieId','rating']].to_csv("entrega_corregida_pablo.csv",index=False)

test_merge_2['rating'].max()
test_merge_2['rating'].min()

entrega_pablo = pd.read_csv("test_groups_pablo.csv",sep = ',')

entrega_pablo[['userId','movieId','rating']].to_csv('entrega_pablo_test.csv',index=False)