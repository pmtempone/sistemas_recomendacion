#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:47:21 2017

@author: pablotempone
"""

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import pandas as pd

#fetch data and format it

train = pd.read_csv("/Users/pablotempone/Google Drive/Maestria/Sistemas de Recomendacion/movilens800k/ratings_train.csv",sep = ',')

#partir en train y test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(train, test_size = 0.1)

data = fetch_movielens(min_rating=4.0)

y = train_df.rating
x = train_df.drop('rating',axis = 1)


print(data['train'])
print(data['test'])

#create model

model = LightFM(loss='warp')



model.fit(data['train'],epochs=30,num_threads=2)


def sample_recommendation(model, data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
            
            
sample_recommendation(model,data,[3,25,450])