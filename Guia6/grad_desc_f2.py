import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def f2(x,y):
    return ((x**2 + y**2)**0.25)*((np.sin(50*(x**2 + y**2)**0.1))**2  + 1)

x = np.linspace(-100, 100, 1000)
y = np.linspace(-100, 100, 1000)

print(f'El mínimo de la función está en ({x[np.argmin(f2(x,y))]},{y[np.argmin(f2(x,y))]},{np.min(f2(x,y))})')

def grad_f2_x(x,y):
    return ((0.5*x*((np.sin(50*(x**2 + y**2)**0.1))**2 + 1))/((x**2+y**2)**0.75) + (20*x*np.cos(50*(x**2 + y**2)**0.1)*np.sin(50*(x**2+y**2)**0.1))/((x**2+y**2)**0.65))
def grad_f2_y(x,y):
    return ((0.5*y*((np.sin(50*(x**2 + y**2)**0.1))**2 + 1))/((x**2+y**2)**0.75) + (20*y*np.cos(50*(x**2 + y**2)**0.1)*np.sin(50*(x**2+y**2)**0.1))/((x**2+y**2)**0.65))

# Parámetros del gradiente descendente
learning_rate = 0.1  # Tasa de aprendizaje
max_iter = 1000     # Número máximo de iteraciones
tolerance = 1e-6      # Tolerancia para la convergencia

# Inicialización aleatoria de x e y
x = np.random.uniform(-100, 100)
y = np.random.uniform(-100, 100)

# Algoritmo del gradiente descendente
for i in range(max_iter):
 # Guardamos los valores anteriores
    x_old = x
    y_old = y

    # Actualizamos x e y usando las derivadas
    x -= learning_rate * grad_f2_x(x, y)
    y -= learning_rate * grad_f2_y(x, y)

    # Criterio de parada: si el cambio es menor que la tolerancia
    if np.sqrt((x - x_old)**2 + (y - y_old)**2) < tolerance:
        print(f'Convergencia alcanzada en la iteración {i}')
        break

# Resultado
minimo = f2(x, y)
print('Mínimo encontrado en (', x, ',', y, ') con valor de la función:', minimo)