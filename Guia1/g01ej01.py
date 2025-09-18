import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math

print(str(datetime.datetime.now()) + " Ejecución iniciada")

### Entrenamiento ###

df_trn = pd.read_csv('Guia1\OR_trn.csv')
mat_trn = df_trn.to_numpy()

# Inicialización de pesos al azar
w = [random.random()-0.5,random.random()-0.5,random.random()-0.5] #[umbral,w_1,w_2]

# Definición de número máximo de etapas
etp_max = 10

# Definición de criterio de finalización
err_max = 0.01

# Definición de tasa de aprendizaje
k = 0.05

# Calcular la cantidad de filas de la matriz
row = mat_trn.shape[0]
x_0 = -np.ones((row,1))
mat_trn = np.hstack((x_0,mat_trn))

# Plotear cada par de puntos x_1, x_2
for i in range(0, row):
    if mat_trn[i][-1]>0: colr = "blue"   # Azul si y=1
    else: colr = "red"              # Rojo si y=-1
    plt.plot(mat_trn[i][1],mat_trn[i][2],marker='o',color=colr,fillstyle="none")
print(str(datetime.datetime.now()) + " Ploteo finalizado")

print(str(datetime.datetime.now()) + " Entrenamiento iniciado")
for etp in range(0,etp_max):
    print(str(datetime.datetime.now()) + " Etapa " + str(etp) + " de " + str(etp_max))
    # Para cada ejemplo de entrenamiento actualizar pesos
    for i in range(0, row):
        # Se obtiene la salida
        y = math.copysign(1,w[0]*mat_trn[i][0]+w[1]*mat_trn[i][1]+w[2]*mat_trn[i][2])
        # Se adaptan los pesos

        for j in range(0,len(w)):
            w[j] = w[j] + ((k/2) * (mat_trn[i][-1]-y)) * mat_trn[i][j]

    err = 0
    # Para cada ejemplo de entrenamiento medir acierto
    for i in range(0, row):
        # Se obtiene la salida
        y = math.copysign(1,w[0]*mat_trn[i][0]+w[1]*mat_trn[i][1]+w[2]*mat_trn[i][2])
        if y != mat_trn[i][-1]:  # Si la salida obtenida es distinta al valor de los datos
            err=err+1       # Se agrega un error al contador
    
    print(str(datetime.datetime.now()) +" "+ str(err) + " errores de " + str(row) + " (" + str((err/row)*100) + "%)")

    if (err/row < err_max): # Si el porcentaje de errores es menor al criterio
        break               # Salir del bucle for

# Print pesos
print(str(datetime.datetime.now()) +" Pesos: u="+ str(w[0]) + " w_1=" + str(w[1]) + " w_2=" + str(w[2]))

# Plotear recta
recta_x1 = [-1.1, 1.1]
recta_x2 = [w[0]/w[2]-recta_x1[0]*(w[1]/w[2]), w[0]/w[2]-recta_x1[1]*(w[1]/w[2])]
plt.plot(recta_x1,recta_x2,color="black")
plt.show()

### Test ###
print(str(datetime.datetime.now()) + " Prueba iniciada")
df_tst = pd.read_csv('Guia1\OR_tst.csv')
mat_tst = df_tst.to_numpy()
rows_tst = mat_tst.shape[0]

y_tst = np.zeros((rows_tst,1))

aciertos = 0

for i in range(0,rows_tst):
    y = math.copysign(1,-w[0]+w[1]*mat_tst[i][0]+w[2]*mat_tst[i][1])
    y_tst[i][0] = y
    if y == mat_tst[i][-1]:
        aciertos = aciertos+1

print(str(datetime.datetime.now()) + " " + str(aciertos) + " aciertos de " + str(rows_tst) + " (" + str((aciertos/rows_tst)*100) + "%)")

print(str(datetime.datetime.now()) + " Ejecución finalizada")