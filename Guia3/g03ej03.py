import numpy as np 

from sklearn import datasets                    # Módulo para levantar los datos
from sklearn.metrics import accuracy_score      # Medida de precisión
from sklearn.model_selection import KFold       # Modelo de partición
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


wine = load_wine()
X, Y = wine.data, wine.target

#USAMOS UN ARBOL DE DECISION COMO CLASIFICADOR
arboldec = DecisionTreeClassifier(max_depth=5)

bagging_clf = BaggingClassifier(arboldec, n_estimators=50)
adaBoost_clf = AdaBoostClassifier(arboldec,n_estimators=50)

# Crear el objeto KFold con 5 particiones
kf_5 = KFold(n_splits=5,shuffle=True,random_state=42)

resultados_bagging = []
resultados_adaBoost = []
for train_i, test_i in kf_5.split(X):
    Xkf5_train , Xkf5_test = X[train_i],X[test_i]
    Ykf5_train , Ykf5_test = Y[train_i],Y[test_i]
    
    bagging_clf.fit(Xkf5_train,Ykf5_train) #Entrenamos
    resultados_bagging.append(bagging_clf.score(Xkf5_test,Ykf5_test))

    adaBoost_clf.fit(Xkf5_train,Ykf5_train) #Entrenamos
    resultados_adaBoost.append(adaBoost_clf.score(Xkf5_test,Ykf5_test))
    
print(f"Precision por cada particion Bagging:, {resultados_bagging}")
mean_bagging=np.mean(resultados_bagging)
var_bagging=np.var(resultados_bagging)
print(f" Bagging- Media: {mean_bagging}, Bagging-Varianza: {var_bagging}")


print(f"Precision por cada particion AdaBoost:, {resultados_adaBoost}")
mean_adaBoost=np.mean(resultados_adaBoost)
var_adaBoost=np.var(resultados_adaBoost)
print(f" Bagging- Media: {mean_adaBoost}, Bagging-Varianza: {var_adaBoost}")
