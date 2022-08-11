#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
df_bucaramanga = pd.read_csv('https://raw.githubusercontent.com/jalvarezp243/practica/master/bucaramanga.csv')
X = df_bucaramanga[['AREA_CONST','ESTRATO','NUM_BANIOS','NUM_PARQ','NUM_HABITA','ANIO_CONST','LATITUD','LONGITUD']]
X = X.values
#Selecciona la variable a predecir
y = df_bucaramanga['VALOR_TOTA']
y = y.values
y = np.log1p(y)
#Se toman los datos de entrenamiento y de testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#Se utiliza el algoritmo Xgboost
model = XGBRegressor()
eval_set = [(X_train, y_train),(X_test, y_test)]
model.fit(X_train, y_train, eval_metric=['rmse'], early_stopping_rounds=10, eval_set=eval_set, verbose=0)
#Precisión del modelo
predict = model.predict(X_test)
print('La precisión del modelo es: ',r2_score(predict, y_test))
#Métricas del módelo
y_prediccion = model.predict(X_test)
print("Error absoluto medio: ", mean_absolute_error(y_test, y_prediccion))
print("Error cuarático medio: ",mean_squared_error(y_test, y_prediccion))
print("R cuadrado: ", r2_score(y_test, y_prediccion))
area_m2 = float(input('Favor ingresar el Área M2: '))
estrato = int(input('Favor ingresar el estrato: '))
num_banios = int(input('Favor ingresar el número de baños: '))
num_parq = int(input('Favor ingresar el número de parqueaderos: '))
num_habit = int(input('Favor ingresar el número de habitaciones: '))
anio_const = int(input('Favor ingresar el año de construcción: '))
latitud = float(input('Favor ingresar la latitud: '))
longitud = float(input('Favor ingresar la longitud: '))
avaluo=pd.DataFrame({'AREA_M2':[area_m2],'ESTRATO':[estrato],'NUM_BANIOS':[num_banios],
                     'NUM_PARQ':[num_parq],'NUM_HABITA':[num_habit],'ANIO_CONST':[anio_const],
                     'LATITUD':[latitud],'LONGITUD':[longitud]})
avaluo=avaluo.values
prediccion=np.expm1(model.predict(avaluo))[0]
print("El valor total es: ", prediccion)
print("El valor M2 es: ", prediccion/area_m2)

