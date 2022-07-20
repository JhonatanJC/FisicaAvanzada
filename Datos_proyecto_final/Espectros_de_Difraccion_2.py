# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(username)s
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')


def DataManagement(file):
    Datos = pd.read_csv(file, sep=' , ', engine='python')
    Datos1 = Datos[21:]
    data = np.array(Datos1)
    df = pd.DataFrame([sub[0].split(",") for sub in data])
    df1 = df.rename(columns={0: 'Angle', 1: 'Intensity'})
    df1 = df1.astype({"Angle": float, 'Intensity': float})
    return df1


def PicosMaximos(Dataframe, intervalosDeBusqueda):
    Intensidad = np.array(Dataframe["Intensity"])
    Angulo = np.array(Dataframe["Angle"])
    # Rangos de los angulos donde visualmente se puede observar que hay un maximo
    rangos_de_busqueda = intervalosDeBusqueda
    picos = []
    for rangos in rangos_de_busqueda:
        new_array_inf = np.full(Angulo.shape, rangos[0])
        diff = np.abs(new_array_inf - Angulo)
        argumento_inf = np.argmin(diff)

        new_array_sup = np.full(Angulo.shape, rangos[1])
        diff = np.abs(new_array_sup - Angulo)
        argumento_sup = np.argmin(diff)

        argumento_intensidad_maxima = np.argmax(
            Intensidad[argumento_inf:argumento_sup])

        angulo_maximo = Angulo[argumento_intensidad_maxima + argumento_inf]
        picos.append(angulo_maximo)

    return picos


def parametro_de_red(Angle, indices_de_miler):
    lamda = 1.541  # En Amstrongs
    Sexagesimal_Angle = Angle*np.pi/180
    distancia_interplanar = lamda/(2*np.sin(Sexagesimal_Angle/2))
    
    print(distancia_interplanar)
    Parametro_de_red = np.sqrt(distancia_interplanar**2 * (
        indices_de_miler[0]**2 + indices_de_miler[1]**2 + indices_de_miler[2] ** 2))
    return Parametro_de_red


rangos_de_busqueda = [(25,26), (27,28),(30, 32), (44, 46),(53, 54.5), (55, 57),
                      (65, 67), (74, 76)]


def Resultados(Path_Archivos, rangos_de_busqueda):
    archivos = [os.path.join(dp, f) for dp, dn, filenames in os.walk(
        Path_Archivos) for f in filenames if os.path.splitext(f)[1] == '.csv']
    Todos_los_parametros_red = []
    temperaturas = []
    plt.figure()

    background = 0
    for archivo in archivos:
        name = archivo.split("\\")[1][:-4]
        Datos = DataManagement(archivo)
        print("Picos sin tratamiento terminco " + name)
        Picos = PicosMaximos(Datos, rangos_de_busqueda)
        print(Picos)
        print("La distancia interplanar  y Parametro de red (En Amstrongs) son  recpectivamente: " + name)
        Parametro_de_red = parametro_de_red(Picos[2], [0, 0, 2])
        print(Parametro_de_red)
        Todos_los_parametros_red.append(Parametro_de_red)
        print('')
        try:
            Temperatura = int(name[-3:])
        except:
            Temperatura = 30
        temperaturas.append(Temperatura)

        plt.plot(np.array(Datos.iloc[:, 0]), np.array(
            Datos.iloc[:, 1]) + background, label=name, markersize=80)
        #plt.yscale("log")
        plt.grid()
        plt.title(
            "Difractograma de una muestra de Sal Natural", fontsize=30)
        plt.xlabel('Ángulo de difracción 2$\\theta$', fontsize=30)
        plt.xticks(fontsize=24)
        plt.ylabel("Intensidad(I)", fontsize=30)
        plt.yticks(fontsize=24)
        plt.legend(loc=1, prop={'size': 25}, facecolor="#b5f1d2")
        #background = background * 10
        background = background + 50000
        plt.show(block=False)

    plt.figure()
    plt.plot(temperaturas, Todos_los_parametros_red,
             '*-', color='red', markersize=40)
    plt.grid()
    plt.title("Parámetros de red para cada temperatura", fontsize=30)
    plt.xlabel('Temperatura ($^{\circ}C$)', fontsize=30)
    plt.xticks(fontsize=24)
    plt.ylabel("a ($ ~\AA$)", fontsize=30)
    plt.yticks(fontsize=24)
    plt.show()


Resultados("./", rangos_de_busqueda)
