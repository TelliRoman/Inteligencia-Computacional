# Importar las librerías necesarias
import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def f_x(x,y):
    return (x**2 + y**2)**(0.25) * ((np.sin(50 * (x**2 + y**2)**(0.1))**2)+1)

#print(funcion(bin_dec(poblacion,-255,255,bit)))

def eval_apt(x,y):
    f_apt = -f_x(x,y)
    min = np.min(f_apt)
    max = np.max(f_apt)
    f_apt = (f_apt - min) / (max - min) 
    return f_apt

def ruleta(poblacion,f_apt,n_giros):
    prob = f_apt / np.sum(f_apt)
    #prob_acum = np.array([])
    #for i in range(0,len(aptitud)):
    #    np.append(prob_acum, (np.sum(prob[0:i])))
    prob_acum = np.cumsum(prob,dtype=float)

    # winner_index = np.array([])
    padres_selec = np.zeros((n_giros,np.shape(poblacion)[1],np.shape(poblacion)[2]))

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
    hijos = np.zeros((len(padres)*cruzas,num_bits,np.shape(poblacion)[2]))
    for i in range(0,len(padres)-1,2):
        for j in range(0,cruzas):
            p = np.random.random()
            if (p <= 0.9): #Si es menor a 0.9 se cruza, sino no
                punto_cruza =  np.random.randint(1,num_bits)
                hijos[i+j*2,:,0] = np.hstack((padres[i,0:punto_cruza,0],padres[i+1,punto_cruza:,0]))
                hijos[i+j*2+1,:,0] = np.hstack((padres[i+1,0:punto_cruza,0],padres[i,punto_cruza:,0]))
                
                hijos[i+j*2,:,1] = np.hstack((padres[i,0:punto_cruza,1],padres[i+1,punto_cruza:,1]))
                hijos[i+j*2+1,:,1] = np.hstack((padres[i+1,0:punto_cruza,1],padres[i,punto_cruza:,1]))
            else:
                hijos[i+j*2] = padres[i]
                hijos[i+j*2+1] = padres[i+1]
    return hijos

def mutacion(hijos):
    for i in range(0,len(hijos)):
        p = np.random.random()
        if (p <= 0.1): 
            mut_punto = np.random.randint(0,num_bits)
            mut_dim = np.random.randint(0,1)
            if(hijos[i][mut_punto][mut_dim] == 0):
                hijos[i][mut_punto][mut_dim] = 1
            else:
                hijos[i][mut_punto][mut_dim] = 0

    return hijos

def bin_dec(poblacion,alpha,beta):
    num_bits = np.shape(poblacion)[1]
    vector_ent = np.flip(np.array([2 ** np.arange(num_bits)]).T)

    d_x = np.dot(poblacion[:,:,0],vector_ent)
    x = alpha + d_x * ((beta-alpha) / ((2**num_bits) - 1))

    d_y = np.dot(poblacion[:,:,1],vector_ent)
    y = alpha + d_y * ((beta-alpha) / ((2**num_bits) - 1))
    return np.hstack((x,y))


#Representacion de los individuos
num_bits = 20
maxit = 1000

size_pob = 21
dim = 2

alpha = -100
beta = 100

poblacion = np.random.choice([0,1],size=(size_pob,num_bits,dim))
poblacion_dec = bin_dec(poblacion,alpha,beta)

# Plotear la función
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.linspace(alpha, beta, 500)
Y = np.linspace(alpha, beta, 500)
X, Y = np.meshgrid(X, Y)
Z = f_x(X,Y)
surf = ax.plot_surface(X, Y, Z, cmap=cm.gray, linewidth=0.5, antialiased=False)
# cm.jet

plt.figure()
plt.pcolor(X, Y, Z, cmap=cm.gray)
plt.scatter(poblacion_dec[:,0],poblacion_dec[:,1],color='red',marker='x')

# Plotear los individuos
z_scat = f_x(poblacion_dec[:,0],poblacion_dec[:,1])
ax.scatter3D(poblacion_dec[:,0],poblacion_dec[:,1],z_scat,color='red',marker='x')

aptitud = eval_apt(poblacion_dec[:,0],poblacion_dec[:,1])

cant_padres = size_pob-1
i = 0
elit = 0
elit_prev = -1
cont = 0
convergencia = 20

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

    aptitud = eval_apt(poblacion_dec[:,0],poblacion_dec[:,1])

    if elit == elit_prev:
        cont += 1
    else:
        elit_prev = elit
        cont = 0
    if (cont == convergencia):
        print('Se llegó a la convergencia en ' + str(i) + ' iteraciones.')
        break

#FALTA COMPARAR CON GRADIENTE DESCENDIENTE Y PROBARLO

# Plotear los individuos de nuevo
plt.scatter(poblacion_dec[:,0],poblacion_dec[:,1],color='green',marker='x')

# Plotear los individuos
z_scat = f_x(poblacion_dec[:,0],poblacion_dec[:,1])
ax.scatter3D(poblacion_dec[:,0],poblacion_dec[:,1],z_scat,color='green',marker='x')

plt.show()