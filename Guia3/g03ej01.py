# Importar las librerías necesarias
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 

# Medidas de desempeño:
from sklearn.metrics import confusion_matrix            # Para generar matriz de confusión
from sklearn.metrics import ConfusionMatrixDisplay


#Usaremos el conjunto de datos Digits que viene con Scikit-learn. 
#Este conjunto de datos contiene imágenes de dígitos escritos a mano.

#Cargar el conjunto de datos digits
digits = load_digits()

#Separar las caracteristicas (X) y las etiquetas (y)
X = digits.data #X son las características (los valores de las imágenes).
Y = digits.target #y son las etiquetas (los dígitos correspondientes).

##Entrenamiento con una única partición (train_test_split)##
# División de los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train , X_test, Y_train , Y_test = train_test_split(X,Y,test_size=0.2)#,random_state=42) #Sacamos random state pq sino siempre genera los mismos aleatorios

#random_state=42 asegura que las divisiones aleatorias sean las mismas cada vez que ejecutes tu código.

#Clasificador de perceptron multicapa
#mlp = MLPClassifier(max_iter=1000,random_state=42) #Predeterminado 1 capa oculta 100 neurnonas, k = 0.001 con un max de 1000 iteraciones Funcion de activacion RELU
mlp = MLPClassifier(hidden_layer_sizes=(10,5),learning_rate_init=0.005, max_iter=1000, activation='logistic')#,random_state=42) #2capas ocultas de 20 y 10 k =0.005 Funcion de activacion Logistic

#Entrenar el modelo
mlp.fit(X_train,Y_train)

# Evaluar el modelo en los datos de prueba
score = mlp.score(X_test, Y_test)
print(f"Tasa de acierto con train_test_split: {score:.4f}")
print()

########################################CREAMOS KFOLD AHORA CON 5 PARTICIONES##################################################
# Crear el objeto KFold con 5 particiones
kf_5 = KFold(n_splits=5,shuffle=True,random_state=42)

# Calcular la media y varianza de los resultados
#scores_5 = cross_val_score(mlp,X,Y,cv=5) #Datos Xdigits Ydigits #Divide en 5, 5 pruebas con 1 particion que es siempre diferente (entrena con la 1ra prueba 2,3,4,5)(entrena con 2da prueba 1,3,4,5)
resultados_kfold = []
for train_i, test_i in kf_5.split(X):
    Xkf5_train , Xkf5_test = X[train_i],X[test_i]
    Ykf5_train , Ykf5_test = Y[train_i],Y[test_i]
    mlp.fit(Xkf5_train,Ykf5_train)
    score = mlp.score(X_test, Y_test)
    resultados_kfold.append(score)

mean_5 = np.mean(resultados_kfold) 
var_5 = np.var(resultados_kfold) 
print(f"Media de la tasa de acierto con 5 particiones (KFold): {mean_5:.4f}")
print(f"Varianza de la tasa de acierto con 5 particiones (KFold): {var_5:.4f}")

# Mostrar la matriz final
print("\nMatriz de resultados KFold (5 particiones):")
for row in resultados_kfold:
     print(row)
print()
########################################CREAMOS KFOLD AHORA CON 10 PARTICIONES##################################################
kf_10 = KFold(n_splits=10,shuffle=True,random_state=42)

# Calcular la media y varianza de los resultados
#scores_5 = cross_val_score(mlp,X,Y,cv=5) #Datos Xdigits Ydigits #Divide en 5, 5 pruebas con 1 particion que es siempre diferente (entrena con la 1ra prueba 2,3,4,5)(entrena con 2da prueba 1,3,4,5)
resultados_kfold = []
for train_i, test_i in kf_10.split(X):
    Xkf10_train , Xkf10_test = X[train_i],X[test_i]
    Ykf10_train , Ykf10_test = Y[train_i],Y[test_i]
    mlp.fit(Xkf10_train,Ykf10_train)
    score = mlp.score(X_test, Y_test)
    resultados_kfold.append(score)

mean_10 = np.mean(resultados_kfold) 
var_10 = np.var(resultados_kfold) 
print(f"Media de la tasa de acierto con 10 particiones (KFold): {mean_10:.4f}")
print(f"Varianza de la tasa de acierto con 10 particiones (KFold): {var_10:.4f}")

# Mostrar la matriz final
print("\nMatriz de resultados KFold (10 particiones):")
for row in resultados_kfold:
     print(row)
