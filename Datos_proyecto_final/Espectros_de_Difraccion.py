# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(username)s
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def DataManagement(file):
    Datos=pd.read_csv(file, sep=' , ')
    Datos1=Datos[21:]
    data= np.array(Datos1)
    df = pd.DataFrame([sub[0].split(",") for sub in data])
    df1 = df.rename(columns={0: 'Angle', 1: 'Intensity'})
    df1 = df1.astype({"Angle": float, 'Intensity': float})
    return df1

def PicosMaximos(Dataframe, intervalosDeBusqueda):
    Intensidad = np.array(Dataframe["Intensity"])
    Angulo = np.array(Dataframe["Angle"])
    rangos_de_busqueda = intervalosDeBusqueda #Rangos de los angulos donde visualmente se puede observar que hay un maximo
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

    return picos

def parametro_de_red(Angle, indices_de_miler):
    lamda = 1.541e-10
    Sexagesimal_Angle = Angle*np.pi/180
    distancia_interplanar =  lamda/(2*np.sin(Sexagesimal_Angle/2))
    Parametro_de_red = np.sqrt(distancia_interplanar**2 * (indices_de_miler[0]**2 + indices_de_miler[1]**2 +indices_de_miler[2]** 2))
    return Parametro_de_red

parametros_de_red = []

Datos_sin_TT = DataManagement("./NaCl_DRX_sin_TT/NaCl_sin_TT.csv")
rangos_de_busqueda_sin_TT = [(11, 12),(23,24),(30, 32), (44,46), (55,57), (65,67), (74,76), (83,85)]
print("Picos sin tratamiento terminco")
Picos_sin_TT = PicosMaximos(Datos_sin_TT, rangos_de_busqueda_sin_TT)
print(Picos_sin_TT)
print("Parametro de red sin tratamiento terminco")
print(parametro_de_red(Picos_sin_TT[2], [0,0,2]))

Datos_300_TT = DataManagement("./NaCl_TT_300/NaCl_TT_300.csv")
rangos_de_busqueda_300_TT = [(30, 32), (44,46), (55,57), (65,67), (74,76), (83,85)]
print("Picos con tratamiento termico de 300")
Picos_300_TT = PicosMaximos(Datos_300_TT, rangos_de_busqueda_300_TT)
print(Picos_300_TT)
print("Parametro de red 300 TT")
print(parametro_de_red(Picos_300_TT[0], [0,0,2]))

Datos_400_TT = DataManagement("./NaCl_TT_400/NaCl_TT_400.csv")
rangos_de_busqueda_400_TT = [(30, 32), (44,46), (55,57), (65,67), (74,76), (83,85)]
print("Picos con tratamiento termico de 400")
Picos_400_TT = PicosMaximos(Datos_400_TT, rangos_de_busqueda_400_TT)
print(Picos_400_TT)
print("Parametro de red 400 TT")
print(parametro_de_red(Picos_400_TT[0], [0,0,2]))

import os
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk("./") for f in filenames if os.path.splitext(f)[1] == '.csv']
print(result)

fig = plt.figure()
plt.plot(np.array(Datos_sin_TT.iloc[:,0]),np.array(Datos_sin_TT.iloc[:,1]), label="Sin tratamiento térmico")
plt.plot(np.array(Datos_300_TT.iloc[:,0]),np.array(Datos_300_TT.iloc[:,1])+50000, label="Tratamiento térmico de 300 $^o$C")
plt.plot(np.array(Datos_400_TT.iloc[:,0]),np.array(Datos_400_TT.iloc[:,1])+100000, label="Tratamiento térmico de 400 $^o$C")

plt.grid()
plt.title("Difractograma de una muestra de Sal Natural", fontsize=30)
plt.xlabel('Ángulo de difracción 2$\\theta$', fontsize=30)
plt.xticks(fontsize=24)
plt.ylabel("Intensidad(I)", fontsize=30)
plt.yticks(fontsize=24)
plt.legend(loc=1, prop={'size': 25},facecolor="#b5f1d2")
plt.show()






