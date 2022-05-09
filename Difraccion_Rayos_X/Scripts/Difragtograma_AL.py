# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Datos=pd.read_csv("Aluminio_grupo5.csv", sep=' , ')
Datos1=Datos[21:]
data= np.array(Datos1)
df = pd.DataFrame([sub[0].split(",") for sub in data])
df1 = df.rename(columns={0: 'Angle', 1: 'Intensity'})
df1 = df1.astype({"Angle": float, 'Intensity': float})

fig = plt.figure()
plt.scatter(np.array(df1.iloc[:,0]),np.array(df1.iloc[:,1]), label="Datos experimentales", s=7)
plt.grid()
plt.title("Difractograma de una muestra de Aluminio", fontsize=30)
plt.xlabel('Ángulo de difracción 2$\\theta$', fontsize=30)
plt.xticks(fontsize=24)
plt.ylabel("Intensidad(I)", fontsize=30)
plt.yticks(fontsize=24)
plt.legend(loc=1, prop={'size': 25},facecolor="#b5f1d2")
plt.show()

Intensidad = np.array(df1["Intensity"])
Angulo = np.array(df1["Angle"])
rangos_de_busqueda = [(37, 41),(42, 47),(63, 68)] #Rangos de los angulos donde visualmente se puede observar que hay un maximo
picos = []
for rangos in rangos_de_busqueda:
    new_array_inf = np.full(Angulo.shape, rangos[0])
    diff = np.abs(new_array_inf - Angulo)
    argumento_inf = np.argmin(diff)
    
    new_array_sup = np.full(Angulo.shape, rangos[1])
    diff = np.abs(new_array_sup - Angulo)
    argumento_sup = np.argmin(diff)

    argumento_intensidad_maxima = np.argmax(Intensidad[argumento_inf:argumento_sup])
    
    angulo_maximo = Angulo[argumento_intensidad_maxima + argumento_inf]
    picos.append(angulo_maximo)

print(picos)