import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
def sigmoide(x,b=5):
    return 2 / (1 + np.exp(-b * np.array(x))) - 1
x=np.linspace(-0.5,0.5,1000)
plt.plot(x,sigmoide(x))
plt.show()
prueba=[0.5 , 0.1 , 0.3 , -0.5]
print(sigmoide(prueba))
'''y = np.array([0.9 ,0 ,-0.6 ])
yd = np.array([1 ,-1 ,1 ])
delta= (yd-y) * (1/2) *((1+y)*(1-y))
def signo(x):
    return np.where(np.array(x) >= 0, 1, -1)
if ((yd == signo(y)).all()):
    print(yd,"   ",signo(y))'''