# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv("./Grupo5_Espectroscopia_Laser_Datos/Absorsion/File_220518_105221.txt", 
                   delimiter = "\t", 
                   header=0,
                   names=["Longitud de Onda(nm)", "Absorbancia"],
                   )

def float_convertion(string_number):
    try:
        return float(string_number)
    except:
        return string_number

def Nan_convertion(number):
    if type(number) != float and type(number) != int:
        return np.NaN
    else:
        return number

Data = Data.reset_index()
Data.drop("index", axis=1, inplace=True)
Data = Data[1:]

#Data = Data.astype({"Absorbancia": float, "Longitud de Onda(nm)": float})#No funciona porque hay valores en Intensidad completados con espacios

Data = Data.applymap(float_convertion) #applymap es solo para todo el dataframe
                                        #map es solo para series o columnas
Data = Data.applymap(Nan_convertion)
print("Valores faltantes")
print(Data.isna().sum())

Data.dropna(subset = ["Absorbancia"], inplace=True)

print("Valores faltantes, despues")
print(Data.isna().sum())

fig = plt.figure()
plt.scatter(np.array(Data.iloc[:,0]),np.array(Data.iloc[:,1]), label="Datos experimentales", s=7)
plt.grid()
plt.title("Espectro de absorción de la muestra $NaYF_{4}$", fontsize=30)
plt.xlabel('Longitud de onda $\lambda$ (nm)', fontsize=30)
plt.xticks(fontsize=24)
plt.ylabel("Absorbancia", fontsize=30)
plt.yticks(fontsize=24)
plt.legend(loc=2, prop={'size': 25},facecolor="#b5f1d2")
plt.show()


Absorbancia = np.array(Data["Absorbancia"])
Longitud_de_onda = np.array(Data["Longitud de Onda(nm)"])

rangos_de_busqueda = [(430, 480), (510, 570), (620, 670), (950, 1000)] #Rangos de los angulos donde visualmente se puede observar que hay un maximo
picos = []
for rangos in rangos_de_busqueda:
    new_array_inf = np.full(Longitud_de_onda.shape, rangos[0])
    diff = np.abs(new_array_inf - Longitud_de_onda)
    argumento_inf = np.argmin(diff)
    
    
    new_array_sup = np.full(Longitud_de_onda.shape, rangos[1])
    diff = np.abs(new_array_sup - Longitud_de_onda)
    argumento_sup = np.argmin(diff)

    argumento_intensidad_maxima = np.argmax(Absorbancia[argumento_inf:argumento_sup])
    
    angulo_maximo = Longitud_de_onda[argumento_intensidad_maxima + argumento_inf]
    picos.append(angulo_maximo)

print("Los picos son los siguientes, respectivamente:")
print(picos)



