#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 23:31:11 2017

@author: pablotempone
"""

import pandas as pd
import numpy as np

restaurantes = pd.read_csv("/Volumes/Disco_SD/Set de datos/guia_oleo/restaurantes.csv",sep = ',',encoding = "ISO-8859-1")

resto_recomendado = restaurantes[['id_restaurante','recomendado_para']]

recomendado = pd.DataFrame(resto_recomendado.recomendado_para.str.split(';'))


recomendado['cleaned'] = np.where(resto_recomendado['recomendado_para'].str.contains('Ir en pareja'), resto_recomendado['recomendado_para'].str.extract('(.*)/)'), resto_recomendado['recomendado_para'])


subs = pd.DataFrame()


subs = subs.where[pd.isnull(resto_recomendado['recomendado_para'].str.find('Ir en pareja'))==False,1]

subs['Ir en pareja']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Ir en pareja'))==False,1,0)
subs['Ir con amigos']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Ir con amigos'))==False,1,0)
subs['Comer con buenos tragos']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer con buenos tragos'))==False,1,0)
subs['Llevar extranjeros']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Llevar extranjeros'))==False,1,0)
subs['Escuchar música']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Escuchar música'))==False,1,0)
subs['Comer sin ser visto']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer sin ser visto'))==False,1,0)
subs['Comer al aire libre']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer al aire libre'))==False,1,0)
subs['Comer solo']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer solo'))==False,1,0)
subs['Reunión de negocios']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Reunión de negocios'))==False,1,0)
subs['Salida de amigas']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Salida de amigas'))==False,1,0)
subs['Comer bien gastando poco']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer bien gastando poco'))==False,1,0)
subs['Ir con la familia']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Ir con la familia'))==False,1,0)
subs['Comer tarde']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer tarde'))==False,1,0)
subs['Comer sano ']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer sano '))==False,1,0)
subs['Merendar']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Merendar'))==False,1,0)
subs['Comer mucho']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Comer mucho'))==False,1,0)
subs['Ir con chicos']= np.where(pd.isnull(resto_recomendado['recomendado_para'].str.find('Ir con chicos'))==False,1,0)
    
