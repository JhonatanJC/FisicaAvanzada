
import numpy as np

import matplotlib.pyplot as plt

 
Temperaturas = [300]
x = np.linspace(0, 200, 100000)

for T in Temperaturas:
    y = np.piecewise(x, [ (x <= T/5) & (x >= 0),(x <= T/5 + 30) & (x >= T/5), (x >= T/5 + 30)], [ lambda x: 5*x, lambda x: T, lambda x:T*1e4*0.90273**(x)])
    #T*1e4*0.90273


    plt.plot(x, y, label='T=300($^{\circ}C$)', markersize=40, color='red')

    plt.grid()
    plt.title("Rampa de temperatura para el calentamiento de la muestra de Sal", fontsize=30)
    plt.xlabel('Tiempo (min)', fontsize=30)
    plt.xticks(fontsize=24)
    plt.ylabel("Temperatura ($^{\circ}C$)", fontsize=30)
    plt.yticks(fontsize=24)
    plt.legend(loc=1, prop={'size': 25},facecolor="#b5f1d2")
plt.show()