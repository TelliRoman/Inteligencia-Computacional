# Importar las librerías necesarias
import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt

def f_x(x):
    return - x * np.sin(np.sqrt(np.abs(x)))

#print(funcion(bin_dec(poblacion,-255,255,bit)))

def eval_apt(x):
    f_apt = -1 * f_x(x)
    min = np.min(f_apt)
    max = np.max(f_apt)
    f_apt = (f_apt - min) / (max - min) 
    return f_apt

def ruleta(poblacion,f_apt,n_giros):
    prob = f_apt  / np.sum(f_apt)
    #prob_acum = np.array([])
    #for i in range(0,len(aptitud)):
    #    np.append(prob_acum, (np.sum(prob[0:i])))
    prob_acum = np.cumsum(prob,dtype=float)

    winner_index = np.array([])
    padres_selec = np.zeros((n_giros,np.shape(poblacion)[1]))

    # Selección sin reemplazo
    """ while len(winner_index) < n_giros:
        r = np.random.random()
        i = np.searchsorted(prob_acum,r)
        if (not np.isin(winner_index, i)):
            np.append(winner_index, i)
            np.vstack(padres_selec,poblacion[i]) """

    # Selección con reemplazo
    for giro in range(0,n_giros-1):
        r = np.random.random()
        i = np.searchsorted(prob_acum,r)
        padres_selec[giro] = np.copy(poblacion[i])

    return padres_selec

#def ventanas(poblacion,aptitud):

def cruza(poblacion,padres,cruzas):
    num_bits = np.shape(poblacion)[1]
    hijos = np.zeros((len(padres)*cruzas,num_bits))
    for i in range(0,len(padres)-1,2):
        for j in range(0,cruzas):
            p = np.random.random()
            if (p <= 0.9): #Si es menor a 0.9 se cruza, sino no
                punto_cruza =  np.random.randint(1,num_bits)
                hijos[i+j*2] = np.hstack((padres[i,0:punto_cruza],padres[i+1,punto_cruza:]))
                hijos[i+j*2+1] = np.hstack((padres[i+1,0:punto_cruza],padres[i,punto_cruza:]))
            else:
                hijos[i+j*2] = padres[i]
                hijos[i+j*2+1] = padres[i+1]
    return hijos

def mutacion(hijos):
    for i in range(0,len(hijos)):
        p = np.random.random()
        if (p <= 0.1): 
            puntomutacion = np.random.randint(0,num_bits)
            if(hijos[i][puntomutacion] == 0):
                hijos[i][puntomutacion] = 1
            else:
                hijos[i][puntomutacion] = 0

    return hijos

def bin_dec(poblacion,alf,beta):
    num_bits = np.shape(poblacion)[1]
    vector_ent = np.flip(np.array([2 ** np.arange(num_bits)]).T)
    d = np.dot(poblacion,vector_ent)
    x = alf + d * ((beta-alf) / ((2**num_bits) - 1))
    return x


#Representacion de los individuos
num_bits = 20
maxit = 100

size_pob = 21

alpha = -255
beta = 255

poblacion = np.random.choice([0,1],size=(size_pob,num_bits))
poblacion_dec = bin_dec(poblacion,alpha,beta)

# Plotear la función
plt.figure(1)
x_graph = np.linspace(alpha,beta,1000)
y_graph = f_x(x_graph)
plt.plot(x_graph,y_graph,color='black')

# Plotear los individuos
y_scat = f_x(poblacion_dec)
plt.scatter(poblacion_dec,y_scat,color='red',marker='x')

aptitud = eval_apt(poblacion_dec)

cant_padres = size_pob-1
elit = 0
elit_prev = -1
cont = 0
convergencia = 10

for i in range(0,maxit):
    padres = ruleta(poblacion,aptitud,cant_padres)
    hijos = cruza(poblacion,padres,1)
    hijos = mutacion(hijos)

    poblacion_new = np.zeros_like(poblacion)
    elit = np.argmax(aptitud)

    poblacion_new[0] = np.copy(poblacion[elit])
    poblacion_new[1:] = np.copy(hijos)
    poblacion = np.copy(poblacion_new)
    poblacion_dec = bin_dec(poblacion,alpha,beta)

    aptitud = eval_apt(poblacion_dec)

    if elit == elit_prev:
        cont += 1
    else:
        elit_prev = elit
        cont = 0
    if (cont == convergencia):
        print('Se llegó a la convergencia en ' + str(i) + ' iteraciones.')
        break

# FALTA COMPARAR CON GRADIENTE DESCENDIENTE Y PROBARLO

# Plotear los individuos de nuevo
y_scat = f_x(poblacion_dec)
plt.scatter(poblacion_dec,y_scat,color='green',marker='x')

plt.show()