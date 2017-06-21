#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:59:36 2017

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

#partir en train y test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(train, test_size = 0.1)

train_avg = train_df[['rating','userID']].groupby('userID',as_index=False).mean()

mean_total = train_df.rating.mean()

train_avg = pd.merge(train_df,train_avg,how='left',left_on='userID',right_on='userID')

train_avg['dif'] = train_avg.rating_x-train_avg.rating_y

test_s = pd.merge(test_df,train_avg,how = 'left',left_on='userID',right_on='userID')

test_s = test_s.fillna(mean_total)


np.round(y_rf,1)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((np.round(y_rf,1) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr_rf.score(x_test, y_test))


def validar_promedios(train_s,test_s):
    train_avg = train_s[['rating','userID']].groupby('userID',as_index=False).mean()
    mean_total = train_s.rating.mean()
    test_s = pd.merge(test_s,train_avg,how = 'left',left_on='userID',right_on='userID')
    test_s = test_s.fillna(mean_total)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((np.round(test_s.rating_y,1) - test_s.rating_x) ** 2))
    

def promedios_movies(train_t,test_t):
    train_avg = train_t[['rating','movieID']].groupby('movieID',as_index=False).mean()
    mean_total = train_t.rating.mean()
    test_s = pd.merge(test_s,train_avg,how = 'left',left_on='userID',right_on='userID')
    test_s = test_s.fillna(mean_total)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((np.round(test_s.rating_y,1) - test_s.rating_x) ** 2))