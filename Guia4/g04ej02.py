import pandas as pd
import random
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
# Comparación con otras etiquetas o generación de matriz de contingencia
from sklearn.metrics.cluster import contingency_matrix
from g04ej01_aux import som_labels

df = pd.read_csv('Guia4/irisbin_trn.csv')
arreglo = df.to_numpy()
row= arreglo.shape[0] # Número de filas del conjunto de datos (número de patrones de entrenamiento)
# Separar las entradas x y los resultados esperados yd
x = arreglo[: , :-3]  #Dejo solo las entradas en una matrix
yd = arreglo[:, -3 :] #Dejo los resultados esperados en un vector
cantentradas = x.shape[1]
k=3

clase_set = arreglo[:,-3]
clase_ver = arreglo[:,-2]
clase_vir = arreglo[:,-1]
color_scat = np.where(clase_set==1,'green',np.where(clase_ver==1,'orange','blue'))
marker_scat = np.where(clase_set==1,'d',np.where(clase_ver==1,'p','v'))

# Inicializar los pesos de las neuronas
#centroide = np.random.random((k,cantentradas))-0.5
centroide = np.zeros((k,cantentradas))
for i in range(k):
    centroide[i] = random.choice(x)
columna = np.full((row,1), -1) 
x = np.hstack((x,columna))
epoc=0

while epoc < 500:
    #print("Época " + str(epoc))
    bandera = False
    for patron in range(0,row):
        pat = x[patron,:-1]
        dif = []
        for i in range(0,k):
            dif.append(pat - centroide[i])
        dist = []
        for i in range(0,k):
            dist.append(np.sqrt(np.dot(dif[i],dif[i])))
        pos=np.argmin(dist)
        if (pos != x[patron,-1]):
            x[patron,-1] = pos
            bandera = True
    if bandera == False:
        break
    for i in range(0,k):
        globals()[f"cent{i}"] = x[x[:, -1] == i]
    for i in range(0,k):
        globals()[f"prom{i}"] = np.mean(globals()[f"cent{i}"], axis=0)
    for i in range(0,k):
        if np.isnan(globals()[f"prom{i}"]).any() == False:
            centroide[i] = globals()[f"prom{i}"][:-1]
    epoc += 1

longS = 0
anchoS = 1
longP = 2
anchoP = 3

plt.figure(1)
plt.scatter(x[:,longS], x[:,anchoS], c=color_scat, marker="o",s=5)
plt.scatter(centroide[:,longS], centroide[:,anchoS],  c="black", marker="D")
plt.title("LongS - AnchoS")
plt.figure(2)
plt.title("LongS - LongP")
plt.scatter(x[:,longS], x[:,longP], c=color_scat, marker="o",s=5)
plt.scatter(centroide[:,longS], centroide[:,longP],  c="black", marker="D")

plt.figure(3)
plt.title("LongS - AnchoP")
plt.scatter(x[:,longS], x[:,anchoP], c=color_scat, marker="o",s=5)
plt.scatter(centroide[:,longS], centroide[:,anchoP],  c="black", marker="D")

plt.figure(4)
plt.title("AnchoS - LongP")
plt.scatter(x[:,anchoS], x[:,longP], c=color_scat, marker="o",s=5)
plt.scatter(centroide[:,anchoS], centroide[:,longP],  c="black", marker="D")

plt.figure(5)
plt.title("AnchoS - AnchoP")
plt.scatter(x[:,anchoS], x[:,anchoP], c=color_scat, marker="o",s=5)
plt.scatter(centroide[:,anchoS], centroide[:,anchoP],  c="black", marker="D")

plt.figure(6)
plt.title("LongP - AnchoP")
plt.scatter(x[:,longP], x[:,anchoP], c=color_scat, marker="o",s=5)
plt.scatter(centroide[:,longP], centroide[:,anchoP],  c="black", marker="D")

print("Clasificacion del K-medias",(x[x[:, -1] == 0]).shape[0],(x[x[:, -1] == 1]).shape[0],(x[x[:, -1] == 2]).shape[0])
print("Clasificacion correcta: [34 32 45]")

plt.show()

# Llamar a la función som_labels()
labels_som,Wx,Wy = som_labels()

# Usar las etiquetas generadas para análisis o comparación
print("Etiquetas generadas por SOM:", labels_som)
matrix = contingency_matrix(x[:, -1], labels_som)
print("Matriz de contingencia :\n", matrix)

