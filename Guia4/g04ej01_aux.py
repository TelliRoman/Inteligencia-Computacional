# Importar las librerías necesarias
import pandas as pd
import random
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt

def som_labels():
    # Funcion que a partir de la posición de la neurona ganadora devuelve una matriz booleana con los vecinos afectados
    def vecinos_mat(G_y,G_x,row,col,vec):
        G_mat = np.zeros((row,col))
        if G_x<0 or G_x>=col or G_y<0 or G_y>=row:
            return G_mat
        for i in range(0,vec+1):
            y_upp = G_y - vec + i
            y_low = G_y + vec - i

            x_lef = G_x - i
            if x_lef<0:
                x_lef = 0

            x_rig = G_x + i
            if x_rig >= col:
                x_rig = col-1

            if y_upp>=0:
                G_mat[y_upp,x_lef:x_rig+1] = 1

            if y_low<row:
                G_mat[y_low,x_lef:x_rig+1] = 1
        return G_mat

    # Función para la distancia de vecinos dependiendo de la etapa
    def vec_fun(epc):
        if epc<50:
            return 2
        elif epc<200:
            return 1
        else:
            return 0

    # Función para la constante de aprendizaje dependiendo de la etapa
    def mu_fun(epc):
        if epc<50:
            return 0.8
        elif epc<200:
            return 1.15 - 0.0035*float(epc)
        else:
            return 0.1

    df = pd.read_csv("Guia4/irisbin_trn.csv")
    mat_datos = df.to_numpy()
    pat_x, pat_y = mat_datos[:,0],mat_datos[:,1]
    pat_row = len(pat_x)

    # Definir constantes
    epc_max = 300
    #mu = 0.2

    # Definir arquitectura (distancia Manhattan)
    row = 7
    col = 7
    #vec = 2

    # Inicializar los pesos de las neuronas
    neu_x = np.random.random((row,col))-0.5
    neu_y = np.random.random((row,col))-0.5

    # Inicializar etiquetas SOM
    som_labels = np.zeros(pat_row)

    # Recorro épocas
    for epc in range(0,epc_max):
        vec = vec_fun(epc)
        mu = mu_fun(epc)
        #print("Época " + str(epc) + " de " + str(epc_max) + ", vec=" + str(vec) + ", mu=" + str(mu))
        # Recorro patrones
        for pat in range(0,pat_row):
            # Diferencias en x y en y para calcular la norma (distancia)
            dif_x = -(neu_x - pat_x[pat])
            dif_y = -(neu_y - pat_y[pat])
            dist = np.sqrt(dif_x * dif_x + dif_y * dif_y)

            # Posición del elemento ganador G y descomposición en fila y columna
            argmin_dist = np.argmin(dist)
            G_row = np.floor_divide(argmin_dist,row)
            G_col = np.mod(argmin_dist,col)

            # Guardar la etiqueta (índice de la neurona ganadora)
            som_labels[pat] = argmin_dist

            # Actualizar los pesos
            # función que devuelve una matriz donde 1 son los vecinos afectados y 0 son los elementos no afectados
            if row==1:
                G_row=0
            if col==1:
                G_col=0

            G_mat = vecinos_mat(G_row,G_col,row,col,vec)
            neu_x = neu_x + mu * (G_mat * dif_x)
            neu_y = neu_y + mu * (G_mat * dif_y)
    return som_labels,neu_x,neu_y
