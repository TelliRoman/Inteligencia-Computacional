# Importar las librerías necesarias
import pandas as pd
import random
import numpy as np
import math
def funcion(x):
    return -x * np.sin(np.sqrt(np.abs(x)))
#print(funcion(bin_dec(poblacion,-255,255,bit)))
def evalucion(x):
    f = -1 * funcion(x)
    min = np.min(f)
    max = np.max(f)
    f = (f - min) / (max - min) 
    return f
def ruleta(poblacion,aptitud,cantvueltas):
    prob = aptitud  / np.sum(aptitud)
    probacum = np.array([])
    for i in range(0,len(aptitud)):
        np.append(probacum, (np.sum(prob[0:i])))
    seleccionados = np.array([])
    for giro in range(0,cantvueltas):
        r = np.random.random()
        for i in range(len(probacum),0):
            if (r <= probacum[i]): 
                np.append(seleccionados, poblacion[i])
                break
    return seleccionados
#def ventanas(poblacion,aptitud):
def cruza(padres):
    for i in range(0,len(padres)-1,2):
        p = np.random.random()
        if (p <= 0.9): #Si es menor a 0.9 se cruza, sino no
            puntocruza =  np.random.randint(1,bit)
            padres[i] = np.hstack(padres[i][0:puntocruza],padres[i+1][puntocruza:])
            padres[i+1] = np.hstack(padres[i+1][0:puntocruza],padres[i][puntocruza:])
    return padres
def mutacion(hijos):
    for i in range(0,len(hijos)):
        p = np.random.random()
        if (p <= 0.1): 
            puntomutacion = np.random.randint(0,bit)
            if(hijos[i][puntomutacion] == 0):
                hijos[i][puntomutacion] = 1
            else:
                hijos[i][puntomutacion] = 0
    return hijos   
def bin_dec(poblacion,alf,beta,bit):
    vector_ent = np.flip(np.array([2 ** np.arange(bit)]).T)
    d = np.dot(poblacion,vector_ent)
    x = alf + d * ((beta-alf) / ((2**bit) - 1))
    return x
#Representacion de los individuos
bit = 10
maxit = 100
sizepoblacion = 111
alf= -255
beta = 255
poblacion_bin = np.random.choice([0,1],size=(sizepoblacion,bit))
poblacion = bin_dec(poblacion_bin,alf,beta,bit)
aptitud = evalucion(poblacion)
cantpadres = sizepoblacion-1
i = 0
elitismo = 0
elitismoant = -1
cont = 0
convergencia = 10
while i < maxit:
    padres=ruleta(poblacion,aptitud,cantpadres) #OPERADOR DE SELECCION
    hijos=cruza(padres) #OPERADOR DE CRUZA
    hijosmutados=mutacion(hijos) #OPERADOR DE MUTACION
    nuevapoblacion = np.zeros(len(poblacion))
    elitismo = np.argmax(aptitud)
    np.append(nuevapoblacion,poblacion[elitismo])
    np.append(nuevapoblacion,hijosmutados)
    poblacion = nuevapoblacion
    aptitud = evalucion(poblacion)
    if elitismo == elitismoant:
        cont += cont
    else:
        elitismoant = elitismo
        cont = 0
    if (cont == convergencia):
        break
    i += 1
#FALTA COMPARAR CON GRADIENTE DESCENDIENTE Y PROBARLO
print(poblacion)