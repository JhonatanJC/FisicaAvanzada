# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data = pd.read_csv("./Grupo5_Espectroscopia_Laser_Datos/Emision/Datos emision.csv", sep='delimiter', header=None)
def Nan_convertion(string):
    if "," in string:
        return string
    else:
        return np.NaN

def float_convertion(string_number):
    try:
        return float(string_number)
    except:
        return string_number

def Nan_convertion_2(number):
    if type(number) != float and type(number) != int:
        return np.NaN
    else:
        return number


Data = Data.applymap(Nan_convertion)
Data.dropna(axis=0, inplace=True)
Data[['Logitud de onda(nm)', 'Intensidad']] = Data[0].str.split(',', expand=True)
Data.drop(0, axis=1, inplace=True)
Data = Data.applymap(float_convertion)
Data = Data.applymap(Nan_convertion_2)
Data.dropna(axis=0, inplace=True)

fig = plt.figure()
plt.plot(np.array(Data.iloc[:,0]),np.array(Data.iloc[:,1]), linewidth=3, label="Datos experimentales")
plt.grid()
plt.title("Espectro de emisión de la muestra $NaYF_{4}$", fontsize=30)
plt.xlabel('Longitud de onda $\lambda$ (nm)', fontsize=30)
plt.xticks(fontsize=24)
plt.ylabel("Absorbancia", fontsize=30)
plt.yticks(fontsize=24)
plt.legend(loc=2, prop={'size': 25},facecolor="#b5f1d2")
plt.show()

#PARA CORTAR LA IMAGEN

Split_Data = Data.loc[(Data['Logitud de onda(nm)'] > 500) & (Data['Logitud de onda(nm)'] < 850)] #Usar | para or y & para and

fig = plt.figure()
plt.plot(np.array(Split_Data.iloc[:,0]),np.array(Split_Data.iloc[:,1]), linewidth=3, label="Datos experimentales")
plt.grid()
plt.title("Espectro de emisión de la muestra $NaYF_{4}$", fontsize=30)
plt.xlabel('Longitud de onda $\lambda$ (nm)', fontsize=30)
plt.xticks(fontsize=24)
plt.ylabel("Intensidad", fontsize=30)
plt.yticks(fontsize=24)
plt.legend(loc=1, prop={'size': 25},facecolor="#b5f1d2")
plt.show()

Intensidad = np.array(Split_Data["Intensidad"])
Longitud_de_onda = np.array(Split_Data["Logitud de onda(nm)"])

rangos_de_busqueda = [(520, 565), (630, 660), (730, 760), (790, 830)] #Rangos de los angulos donde visualmente se puede observar que hay un maximo
picos = []
for rangos in rangos_de_busqueda:
    new_array_inf = np.full(Longitud_de_onda.shape, rangos[0])
    diff = np.abs(new_array_inf - Longitud_de_onda)
    argumento_inf = np.argmin(diff)
    
    new_array_sup = np.full(Longitud_de_onda.shape, rangos[1])
    diff = np.abs(new_array_sup - Longitud_de_onda)
    argumento_sup = np.argmin(diff)

    argumento_intensidad_maxima = np.argmax(Intensidad[argumento_inf:argumento_sup])
    
    angulo_maximo = Longitud_de_onda[argumento_intensidad_maxima + argumento_inf]
    picos.append(angulo_maximo)

print(picos)


