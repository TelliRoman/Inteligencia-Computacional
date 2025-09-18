import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return -x*np.sin(np.sqrt(abs(x)))
def f_deri(x):
    return -np.sin(np.sqrt(abs(x))) - (x**2 * np.cos(np.sqrt(abs(x))))/(2*abs(x)*np.sqrt(abs(x)))

# Parámetros del gradiente descendiente
learning_rate = 0.1   # Constante de aprendizaje
x = np.random.randint(-512, 512)  # Inicializamos en un número aleatorio del dominio
max_iter = 10000        # Máximo número de iteraciones
tolerance = 1e-6       # Tolerancia para la convergencia

# Algoritmo del gradiente descendiente
for i in range(max_iter):
    gradiente = f_deri(x)
    nuevo_x = x - learning_rate * gradiente

    # Criterio de parada: Si el cambio es menor que la tolerancia, detener
    if np.abs(nuevo_x - x) < tolerance:
        break

    x = nuevo_x  # Actualizamos x

# Resultado
print('Mínimo encontrado en (', x, ',', f(x), ')')