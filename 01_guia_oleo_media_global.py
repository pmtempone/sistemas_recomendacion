#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:23:30 2017

@author: pablotempone
"""

#primer modelo - mean

import pandas as pd
import numpy as np

train = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/ratings_train.csv",sep = ',',encoding = "ISO-8859-1")

#separar en train y test

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(train, test_size = 0.03)

global_mean_ambiente = train_df.rating_ambiente.mean()

global_mean_comida = train_df.rating_comida.mean()

global_mean_servicio = train_df.rating_servicio.mean()

test_df['pred_ambiente'] = global_mean_ambiente

print("Mean squared error ambiente: %.2f"
      % np.mean((np.round(test_df.pred_ambiente,1) - test_df.rating_ambiente) ** 2))

test_df['pred_comida'] = global_mean_comida

print("Mean squared error comida: %.2f"
      % np.mean((np.round(test_df.pred_comida,1) - test_df.rating_comida) ** 2))

test_df['pred_servicio'] = global_mean_servicio

print("Mean squared error servicio: %.2f"
      % np.mean((np.round(test_df.pred_servicio,1) - test_df.rating_servicio) ** 2))

print("Mean squared error Global: %.2f"
      % np.mean((np.round(test_df.pred_servicio+test_df.pred_comida+test_df.pred_ambiente,1) - (test_df.rating_servicio+test_df.rating_ambiente+test_df.rating_comida)) ** 2))


