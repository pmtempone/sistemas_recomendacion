#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 19:08:23 2017

@author: pablotempone
"""
import pandas as pd
import psycopg2
import matplotlib as mp
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn import linear_model



train = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_train.csv",sep = ',')

movie_actors = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_actors.csv",sep = ',')
movie_countries = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_countries.csv",sep = ',')
movie_directors = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_directors.csv",sep = ',')
movie_genres = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_genres.csv",sep = ',')
movie_imdb = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_imdb.csv",sep = ',')
movie_locations = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/movie_locations.csv",sep = ',')


plt.hist(train['rating'], bins=10)
