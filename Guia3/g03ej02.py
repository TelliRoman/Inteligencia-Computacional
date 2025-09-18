import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#Cargar el conjunto de datos digits
digits = load_digits()

#Separar las caracteristicas (X) y las etiquetas (y)
X = digits.data #X son las características (los valores de las imágenes).
Y = digits.target #y son las etiquetas (los dígitos correspondientes).

mlp = MLPClassifier(hidden_layer_sizes=(10,5),learning_rate_init=0.005, max_iter=1000, activation='logistic')#,random_state=42) #2capas ocultas de 20 y 10 k =0.005 Funcion de activacion Logistic
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 5)
arboldec = DecisionTreeClassifier()
soportevec = SVC()
lda = LinearDiscriminantAnalysis()
# Crear el objeto KFold con 5 particiones
kf_5 = KFold(n_splits=5,shuffle=True,random_state=42)

resultados_mlp = []
resultados_gnb = []
resultados_knn = []
resultados_arboldec = []
resultados_soportevec = []
resultados_adl = []
for train_i, test_i in kf_5.split(X):
    Xkf5_train , Xkf5_test = X[train_i],X[test_i]
    Ykf5_train , Ykf5_test = Y[train_i],Y[test_i]
    
    mlp.fit(Xkf5_train,Ykf5_train)
    resultados_mlp.append(mlp.score(Xkf5_test, Ykf5_test))
    
    gnb.fit(Xkf5_train,Ykf5_train)
    ygnb_pred = gnb.predict(Xkf5_test)
    resultados_gnb.append(accuracy_score(Ykf5_test,ygnb_pred))

    knn.fit(Xkf5_train,Ykf5_train)
    resultados_knn.append(knn.score(Xkf5_test, Ykf5_test))

    arboldec.fit(Xkf5_train,Ykf5_train)
    resultados_arboldec.append(arboldec.score(Xkf5_test, Ykf5_test))

    soportevec.fit(Xkf5_train,Ykf5_train)
    resultados_soportevec.append(soportevec.score(Xkf5_test, Ykf5_test))
    
    lda.fit(Xkf5_train,Ykf5_train)
    resultados_adl.append(lda.score(Xkf5_test, Ykf5_test))
    
mean_mlp = np.mean(resultados_mlp) 
var_mlp = np.var(resultados_mlp) 

mean_gnb = np.mean(resultados_gnb) 
var_gnb = np.var(resultados_gnb) 

mean_knn = np.mean(resultados_knn) 
var_knn = np.var(resultados_knn) 

mean_arboldec = np.mean(resultados_arboldec) 
var_arboldec = np.var(resultados_arboldec) 

mean_svc = np.mean(resultados_soportevec) 
var_svc = np.var(resultados_soportevec) 

mean_lda = np.mean(resultados_adl) 
var_lda = np.var(resultados_adl) 

print(f"MLP - Media: {mean_mlp}, Varianza: {var_mlp}")
print(f"GNB - Media: {mean_gnb}, Varianza: {var_gnb}")
print(f"KNN - Media: {mean_knn}, Varianza: {var_knn}")
print(f"Árbol de Decisión - Media: {mean_arboldec}, Varianza: {var_arboldec}")
print(f"SVC - Media: {mean_svc}, Varianza: {var_svc}")
print(f"LDA - Media: {mean_lda}, Varianza: {var_lda}")