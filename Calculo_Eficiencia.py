import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

def func(x, a, b, c):
    return a * np.exp(b * x) + c
def func2(x, a, b, c):
    return a*x**2 + b*x + c

def funcPotencia(x, a, b, c):
    return (a * np.exp(b * x) + c)*x

# R en Ohm
R = [15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000, 1500]
# V en V
V = [0, 11.43, 18.98, 20.96, 21.15, 21.29, 21.33, 21.53, 21.64, 21.72, 21.81, 21.87, 21.96, 21.93]
# I en A
I = [0.39, 0.39, 0.3, 0.19, 0.16, 0.14, 0.13, 0.01, 0.08, 0.06, 0.04, 0.03, 0.01, 0]

V = np.array(V)
I = np.array(I)

plt.figure()

popt, pcov = curve_fit(func, np.array(V), np.array(I))
residuals = I - func(V, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((I-np.mean(I))**2)
r_squared = 1 - (ss_res / ss_tot)
print("La constante R2 del ajuste realizado es:")
print(r_squared)

parameters=[]
for index,i in enumerate(popt):
    parameters.append(i)
    parameters.append(pcov[index,index]**0.5)
parameters.append(r_squared)
print(parameters)
V_fit = np.linspace(0,22,1000)

plt.scatter(V, I, label="Datos experimentales", s=90)
plt.plot(V_fit, func(V_fit, *popt), 'r-', label="$I(V)=a.e^{b.V}+c$"+"\n" +"fit: a=%f $\pm$ %5.5f,\n b=%5.3f $\pm$ %5.3f,\n c=%5.3f $\pm$ %5.3f \n $R^{2}$=%5.3f" % tuple(parameters))
plt.grid()
plt.title("Gráfica experimental I(A) vs V(V)", fontsize=30)
plt.xlabel("Voltaje V(V)", fontsize=30)
plt.xticks(fontsize=24)
plt.ylabel("Corriente I(A)", fontsize=30)
plt.yticks(fontsize=24)
plt.legend(loc=3, prop={'size': 25},facecolor="#b5f1d2")
plt.show()

P = np.array(V) * np.array(I)

plt.figure()
plt.scatter(V, P, label="Datos experimentales", s=90)
plt.plot(V_fit, funcPotencia(V_fit, *popt), 'r-', label="$P(V) = I(V)*V = (a.e^{b.V}+c)*V$"+ "\n" +"fit: a=%5.6f $\pm$ %5.5f,\n b=%5.3f $\pm$ %5.3f,\n c=%5.3f $\pm$ %5.3f \n $R^{2}$=%5.3f" % tuple(parameters))
plt.grid()
plt.title("Gráfica experimental P(W) vs V(V)", fontsize=30)
plt.xlabel("Voltaje V(V)", fontsize=30)
plt.xticks(fontsize=24)
plt.ylabel("Potencia P(W)", fontsize=30)
plt.yticks(fontsize=24)
plt.legend(loc=2, prop={'size': 25},facecolor="#b5f1d2")
plt.show()

x0 = [1]#PUNTO INCIAL DE REFERENCIA PARA EL ALGORITMO
#LA FUNCION LAMBDA TIENE SIGNO NEGATIVO DEBIDOD A QUE SE QUIERE CALCULAR EL MÁXIMO Y NO EL MÍNIMO
res = minimize(lambda x: -funcPotencia(x,*popt),x0, method='Nelder-Mead', tol=1e-6)
print("Entonces el punto en que la funcion P(V) es máximo es (en V):")
print(res.x)
print("Por lo tanto la pontencia máxima es (en W):")
print(abs(res.fun))
