import pandas as pd
import random
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
def signo(x):
    if(x<0): return -1
    else: return 1
def sigmoide(x,b=1):
    return 2 / (1 + np.exp(-b * np.array(x))) - 1
def ej1(Estructura,str_tst,str_trn): #[2 1]
    df = pd.read_csv(str_trn)
    arreglo = df.to_numpy()
    row= arreglo.shape[0] # Número de filas del conjunto de datos (número de patrones de entrenamiento)

    # Separar las entradas x y los resultados esperados yd
    x = arreglo[: , :-1]  #Dejo solo las entradas en una matrix
    columna = np.full((arreglo.shape[0],1), -1) #Creo una columna de solo -1 para el sesgo
    x = np.hstack((columna,x)) #Agrego la s columnas de -1 a las entradas
    yd = arreglo[:,-1] #Dejo los resultados esperados en un vector
    
    cantEntradas = x.shape[1]  # Cantidad de columnas de la matriz de entrada (entradas + sesgo)
    Capas = Estructura.shape[0]  # Número de capas de la red (definida por el array `Estructura`)

    # Inicialización de listas para almacenar pesos, salidas, y deltas para cada capa
    ListaW = []  # Lista de matrices de pesos para cada capa
    ListaE = []  # Lista de salidas (outputs) para cada capa
    ListaDelta = []  # Lista de errores (deltas) para cada capa

    for capa in range(0,Capas):
        if capa == 0:
                ListaW.append(np.random.uniform(low=-0.5,high=0.5,size=(Estructura[capa],cantEntradas)))
        else:
            ListaW.append(np.random.uniform(low=-0.5,high=0.5,size=(Estructura[capa],Estructura[capa-1]+1)))
        ListaE.append(np.zeros((Estructura[capa])))
        ListaDelta.append(np.zeros((Estructura[capa])))
   
    epoca_max = 200
    err_max = 0.01
    k = 0.01
   
    # Plotear cada par de puntos x_1, x_2
    for i in range(0, row):
        if arreglo[i][-1]>0: colr = "blue"   # Azul si y=1
        else: colr = "red"              # Rojo si y=-1
        plt.plot(arreglo[i][0],arreglo[i][1],marker='o',color=colr,fillstyle="none")
    
    # Entrenamiento de la red neuronal
    for epoca in range(0,epoca_max):
        for i in range(0,row):
            entrada=x[i]  # Obtener la entrada actual          
            for capa in range(0,Capas):
                #PROPAGACION HACIA ADELANTE
                W = ListaW[capa] #Tengo los W asociados a la capa
                aux = (np.dot(W,entrada))
                ListaE[capa]= sigmoide(aux) #Matriz de 2x3 y vector de 3x1
                entrada = np.hstack((-1,ListaE[capa])) #Le agrego el -1 para la entrada del sesgo de la capa siguiente          
            #PROPAGACION HACIA ATRAS, capa final
            y=ListaE[-1]  #Programado para que sea un numero, hay q programar para que sea una lista
            ListaDelta[-1] =(yd[i]-y) * (1/2) *((1+y)*(1-y))
            # Cálculo de deltas para capas ocultas
            for capa in range(Capas-1,0,-1):
                y_ant=ListaE[capa-1] # Salida de la capa anterior
                w=ListaW[capa] # Pesos de la capa actual
                w=w[:,1:].T  # Excluir el peso del sesgo #Todo los renglones que son las neuranos y todas las columnas menos la 1 que es el peso des umbral
                delta = np.dot(w,ListaDelta[capa]) # Cálculo del delta de la capa actual
                ListaDelta[capa-1] = delta * (1/2)*((1+y_ant)*(1-y_ant))
            #ACTUALIZAR LOS PESOS
            entrada=x[i] # Reiniciar entrada
            for capa in range(0,Capas):
               #ListaW[capa] = ListaW[capa] + k * ListaDelta[capa].reshape(-1,1) @ entrada.reshape(-1,1)
               # Actualización de los pesos usando los deltas calculados
                ListaW[capa] = ListaW[capa] + k * np.outer(ListaDelta[capa], entrada)
                entrada = np.hstack((-1, ListaE[capa]))  # Agregar -1 para el sesgo de la siguiente capa

        # Evaluación del rendimiento de la red después de cada época 
        # Para cada ejemplo de entrenamiento (patron) medir acierto
        err = 0
        for i in range(0,row):
            entrada=x[i]  # Reiniciar entrada
            for capa in range(0,Capas):
                #PROPAGACION HACIA ADELANTE
                W = ListaW[capa] #Tengo los W asociados a la capa
                ListaE[capa]= sigmoide((np.dot(W,entrada))) #Matriz de 2x3 y vector de 3x1
                entrada = np.hstack((-1,ListaE[capa])) #Le agrego el -1 para la entrada del sesgo de la capa siguiente
            
            ysign= signo(ListaE[-1])  # Obtener el signo de la salida final
            if (yd[i] != ysign):
                err += 1
      
        print(str(datetime.datetime.now()) +" "+ str(err) + " errores de " + str(row) + " (" + str((err/row)*100) + "%)")
        if (err/row < err_max): # Si el porcentaje de errores es menor al criterio
            break               # Salir del bucle for
  
    # Imprimir los pesos finales
    print("Los pesos son:", ListaW)
    
    # Visualización de las rectas de decisión
    w = ListaW[0][0]
    recta_x1 = [-1.1, 1.1]
    recta_x2 = [w[0]/w[2] - recta_x1[0] * (w[1]/w[2]), w[0]/w[2] - recta_x1[1] * (w[1]/w[2])]
    plt.plot(recta_x1, recta_x2, color="black")
    w = ListaW[0][1]
    recta_x1 = [-1.1, 1.1]
    recta_x2 = [w[0]/w[2] - recta_x1[0] * (w[1]/w[2]), w[0]/w[2] - recta_x1[1] * (w[1]/w[2])]
    plt.plot(recta_x1, recta_x2, color="black")
    plt.show()
    
    # Prueba con el conjunto de datos de prueba
    print(str(datetime.datetime.now()) + " Prueba iniciada")
    df_tst = pd.read_csv(str_tst)  # Cargar el conjunto de prueba
    mat_tst = df_tst.to_numpy()  # Convertir los datos de prueba a una matriz NumPy
    rows_tst = mat_tst.shape[0]  # Obtener el número de patrones de prueba

    x = mat_tst[:, :-1]  # Dejar solo las entradas en una matriz
    columna = np.full((mat_tst.shape[0], 1), -1)  # Crear una columna de solo -1 (para el sesgo)
    x = np.hstack((columna, x))  # Agregar la columna de -1 a las entradas
    yd = mat_tst[:, -1]  # Dejar los resultados esperados en un vector
    y_tst = np.zeros((rows_tst, 1))  # Inicializar el vector de salidas de prueba
    aciertos = 0  # Contador de aciertos

    for i in range(0,rows_tst):
        entrada=x[i]
        for capa in range(0,Capas):
            #PROPAGACION HACIA ADELANTE
            W = ListaW[capa] #Tengo los W asociados a la capa
            ListaE[capa]= sigmoide((np.dot(W,entrada))) #Matriz de 2x3 y vector de 3x1
            entrada = np.hstack((-1,ListaE[capa])) #Le agrego el -1 para la entrada del sesgo de la capa siguiente
       
        y= signo(ListaE[-1])
        y_tst[i][0] = y
       
        if y == yd[i]:
            aciertos = aciertos+1
    
    print(str(datetime.datetime.now()) + " " + str(aciertos) + " aciertos de " + str(rows_tst) + " (" + str((aciertos/rows_tst)*100) + "%)")
    print(str(datetime.datetime.now()) + " Ejecución finalizada")
    return 0
ej1(np.array([2,1]),'Guia1/XOR_tst.csv','Guia1/XOR_trn.csv')