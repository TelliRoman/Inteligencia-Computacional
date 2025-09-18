# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Funciones
def sigmoide(x):
    return -1/(1 + np.exp(-0.01 * (x-500))) + 1

def ruleta(poblacion,f_apt,n_giros):
    prob = f_apt  / np.sum(f_apt)
    prob_acum = np.cumsum(prob,dtype=float)

    winner_index = np.array([])
    padres_selec = np.zeros((n_giros,np.shape(poblacion)[1]))

    # Selección con reemplazo
    for giro in range(0,n_giros-1):
        r = np.random.random()
        i = np.searchsorted(prob_acum,r)
        padres_selec[giro] = np.copy(poblacion[i])

    return padres_selec

def cruza(poblacion,padres,cruzas):
    num_bits = np.shape(poblacion)[1]
    hijos = np.zeros((len(padres)*cruzas,num_bits))
    for i in range(0,len(padres)-1,2):
        for j in range(0,cruzas):
            p = np.random.random()
            hijos[i+j*2] = np.copy(padres[i])
            hijos[i+j*2+1] = np.copy(padres[i+1])
            if (p <= 0.9): #Si es menor a 0.9 se cruza, sino no
                punto_cruza =  np.random.randint(1,num_bits-1)
                hijos[i+j*2,0:punto_cruza] = np.copy(padres[i+1,0:punto_cruza])
                hijos[i+j*2+1,0:punto_cruza] = np.copy(padres[i,0:punto_cruza])
    
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

# Importar datos
df = pd.read_csv("Guia6/leukemia_train.csv",header=None)
mat_trn = df.to_numpy()

df = pd.read_csv("Guia6/leukemia_test.csv",header=None)
#mat_tst = df.to_numpy()

# Inicializar población
num_bits = 7129
size_pob = 51
poblacion = np.zeros((size_pob,num_bits))
for i in range(0,size_pob):
    cant_rand = np.random.randint(low=30,high=100)
    pos_rand = np.random.randint(low=0,high=num_bits,size=cant_rand)
    poblacion[i,pos_rand] = 1


# Evaluar población
aptitud = np.zeros(size_pob)

indexes_tst = np.sort(np.random.randint(0,37,10))

mat_tst = mat_trn[indexes_tst,:]
mat_trn = np.delete(mat_trn,indexes_tst,axis=0)

for i in range(0,size_pob):
    indexes = np.where(poblacion[i] == 1)[0]

    x_trn = np.array(mat_trn[:,indexes])
    y_trn = np.array(mat_trn[:,-1])
    x_tst = np.array(mat_tst[:,indexes])
    y_tst = np.array(mat_tst[:,-1])

    svc_class = SVC()
    svc_class.fit(x_trn,y_trn)
    svc_accuracy = svc_class.score(x_tst,y_tst)
    x = len(indexes)
    aptitud[i] = svc_accuracy*sigmoide(x)


#print(aptitud)


# Variables y constantes
cant_padres = size_pob-1
elit = 0
elit_prev = -1
cont = 0
convergencia = 50
maxit = 1000

for it in range(0,maxit):
    #print("It:" + str(it))
    padres = ruleta(poblacion,aptitud,cant_padres)
    hijos = cruza(poblacion,padres,1)
    hijos = mutacion(hijos)

    poblacion_new = np.zeros_like(poblacion)
    elit = np.argmax(aptitud)

    poblacion_new[0] = np.copy(poblacion[elit])
    poblacion_new[1:] = np.copy(hijos)
    poblacion = np.copy(poblacion_new)

    aptitud = np.zeros(size_pob)
    for i in range(0,size_pob):
        indexes = np.where(poblacion[i] == 1)[0]
        if len(indexes>0):
            x_trn = np.array(mat_trn[:,indexes])
            y_trn = np.array(mat_trn[:,-1])
            x_tst = np.array(mat_tst[:,indexes])
            y_tst = np.array(mat_tst[:,-1])

            svc_class = SVC()
            svc_class.fit(x_trn,y_trn)
            svc_accuracy = svc_class.score(x_tst,y_tst)
            x = len(indexes)
            aptitud[i] = svc_accuracy*sigmoide(x)

    if elit == elit_prev:
        cont += 1
    else:
        elit_prev = elit
        cont = 0
    if (cont == convergencia):
        print('Se llegó a la convergencia en ' + str(it) + ' iteraciones.')
        print('El subcojunto de características es: ')
        print(np.where(poblacion[0] == 1)[0])
        print(len(np.where(poblacion[0] == 1)[0]))
        break

    