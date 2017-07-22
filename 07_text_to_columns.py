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



subs['Ir en pareja']= np.where((resto_recomendado['recomendado_para'].str.find('pareja')>=0)==True,1,0)
subs['Ir con amigos']= np.where((resto_recomendado['recomendado_para'].str.find('amigos')>=0)==True,1,0)
subs['Comer con buenos tragos']= np.where((resto_recomendado['recomendado_para'].str.find('tragos')>=0)==True,1,0)
subs['Llevar extranjeros']= np.where((resto_recomendado['recomendado_para'].str.find('extranjeros')>=0)==True,1,0)
subs['Escuchar música']= np.where((resto_recomendado['recomendado_para'].str.find('música')>=0)==True,1,0)
subs['Comer sin ser visto']= np.where((resto_recomendado['recomendado_para'].str.find('visto')>=0)==True,1,0)
subs['Comer al aire libre']= np.where((resto_recomendado['recomendado_para'].str.find('libre')>=0)==True,1,0)
subs['Comer solo']= np.where((resto_recomendado['recomendado_para'].str.find('solo')>=0)==True,1,0)
subs['Reunión de negocios']= np.where((resto_recomendado['recomendado_para'].str.find('negocios')>=0)==True,1,0)
subs['Salida de amigas']= np.where((resto_recomendado['recomendado_para'].str.find('amigas')>=0)==True,1,0)
subs['Comer bien gastando poco']= np.where((resto_recomendado['recomendado_para'].str.find('poco')>=0)==True,1,0)
subs['Ir con la familia']= np.where((resto_recomendado['recomendado_para'].str.find('familia')>=0)==True,1,0)
subs['Comer tarde']= np.where((resto_recomendado['recomendado_para'].str.find('tarde')>=0)==True,1,0)
subs['Comer sano ']= np.where((resto_recomendado['recomendado_para'].str.find('sano ')>=0)==True,1,0)
subs['Merendar']= np.where((resto_recomendado['recomendado_para'].str.find('Merendar')>=0)==True,1,0)
subs['Comer mucho']= np.where((resto_recomendado['recomendado_para'].str.find('mucho')>=0)==True,1,0)
subs['Ir con chicos']= np.where((resto_recomendado['recomendado_para'].str.find('chicos')>=0)==True,1,0)
    
restaurantes_recomendado = pd.concat([restaurantes,subs],axis=1)


resto_medios = restaurantes[['id_restaurante','medios_pago']]

medios = pd.DataFrame(resto_medios.medios_pago.str.split(';',expand=True))

s = pd.Series((resto_medios['medios_pago'].str.split(';')))

medios = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)

restaurantes_recomendado = pd.concat([restaurantes_recomendado,medios],axis = 1)

restaurantes_recomendado.iloc[:,37:46] = restaurantes_recomendado.iloc[:,37:46].fillna(0)

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:pass@localhost:5432/sistemas_recomendacion')

restaurantes_recomendado.to_sql('restaurantes_v2', engine,index=False)
