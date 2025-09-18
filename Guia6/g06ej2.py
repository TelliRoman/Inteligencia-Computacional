import pandas as pd
import random
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

df = pd.read_csv("Guia6\leukemia_train.csv")
arreglo = df.to_numpy()
row = arreglo.shape[0] 
x = arreglo[: , :-1]  
col = x.shape[1]
yd = arreglo[:,-1]

poblacion_bin = np.random.choice([0,1],size=((row-10),col))
idx_pruebainter = np.sort(np.random.randint(0,37,10))
x_pruebainter = x[idx_pruebainter,:]
yd_pruebainter = yd[idx_pruebainter,:] 
x_trn = np.delete(x,idx_pruebainter,axis=0)
yd_trn = np.delete(yd,idx_pruebainter,axis=0)

vec_accuaracy=np.array([])
for indv in range(0,poblacion_bin.shape[0]):    
    indices = np.where(poblacion_bin[indv] == 1)[0]
    x_features = x_trn[:,indices] #Me guardo las columnas en las que aparezca un 1
    soportevec = SVC()
    soportevec.fit(x_features,yd_trn)
    accuaracy=soportevec.score(x_pruebainter,yd_pruebainter)
    np.append(vec_accuaracy,accuaracy)
