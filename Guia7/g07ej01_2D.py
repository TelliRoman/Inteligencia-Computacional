import numpy as np
import time

# Función para inicializar el enjambre aleatorio con velocidades iniciales nulas
def enjambre(num_particulas, mins, maxs):
    posiciones = np.random.uniform(mins, maxs, (num_particulas, len(mins)))  # Posiciones aleatorias en el rango
    velocidades = np.zeros((num_particulas, len(mins)))  # Velocidades iniciales nulas
    mejor_pos_personal = np.copy(posiciones)
    return posiciones, velocidades, mejor_pos_personal

def algoritmo(num_particulas, mins, maxs, funcion, c1, c2, max_iteraciones, tolerancia):
    inicio = time.perf_counter() 
    # Inicializamos el enjambre
    posiciones, velocidades, mejor_pos_personal = enjambre(num_particulas, mins, maxs)
  
    # Evaluamos la función de error en las posiciones iniciales
    valores_error_personal = funcion(posiciones[:, 0], posiciones[:, 1])
    
    # Inicializamos la mejor posición global
    mejor_pos_global = mejor_pos_personal[np.argmin(valores_error_personal)]
    mejor_error_global = np.min(valores_error_personal)

    # Guardar el error global de la iteración anterior para la condición de convergencia
    mejor_error_anterior = mejor_error_global

    c1_inicial, c1_final = c1
    c2_inicial, c2_final = c2

    for iteracion in range(max_iteraciones):
        #print(f"Iteración {iteracion + 1}")
        # Decrementamos c1 e incrementamos c2 por cada iteración
        c1_actual = c1_inicial - (c1_inicial - c1_final) * (iteracion / max_iteraciones)
        c2_actual = c2_inicial + (c2_final - c2_inicial) * (iteracion / max_iteraciones)

        for i in range(num_particulas):
            r1 = np.random.rand()  # Componente aleatoria para el peso personal
            r2 = np.random.rand()  # Componente aleatoria para el peso global
            
            # Actualización de la velocidad (para el caso 2D)
            velocidades[i] = (velocidades[i] + 
                              c1_actual * r1 * (mejor_pos_personal[i] - posiciones[i]) +  # Influencia personal
                              c2_actual * r2 * (mejor_pos_global - posiciones[i]))  # Influencia global
          
            posiciones[i] = posiciones[i] + velocidades[i]
    
            posiciones[i] = np.clip(posiciones[i], mins, maxs)  # Mantener partículas dentro de los rangos permitidos
       
        valores_error_actuales = funcion(posiciones[:, 0], posiciones[:, 1])  # Evaluamos las nuevas posiciones en 2D

        for i in range(num_particulas):  # Actualizamos las mejores posiciones personales
            if valores_error_actuales[i] < valores_error_personal[i]:
                mejor_pos_personal[i] = posiciones[i]
                valores_error_personal[i] = valores_error_actuales[i]

        # Actualizamos la mejor posición global
        mejor_error_iteracion = np.min(valores_error_personal)
        if mejor_error_iteracion < mejor_error_global:
            mejor_error_global = mejor_error_iteracion
            mejor_pos_global = mejor_pos_personal[np.argmin(valores_error_personal)]

        # Comprobación de la condición de convergencia (tolerancia)
        if abs(mejor_error_anterior - mejor_error_global) < tolerancia:
            fin = time.perf_counter()
            print(f'Tiempo de convergencia {fin - inicio} segundos.') 
            print(f"Iteraciones{iteracion + 1}.")
            break    

        # Actualizamos el valor del mejor error anterior para la siguiente iteración
        mejor_error_anterior = mejor_error_global

    return mejor_pos_global

#############################
###### EJEMPLO DE USO ########
#############################

# Función de error para 2D
def f2(x, y):
    return ((x**2 + y**2)**0.25) * ((np.sin(50 * (x**2 + y**2)**0.1))**2 + 1)
x = np.linspace(-100, 100, 1000)
y = np.linspace(-100, 100, 1000)
print(f'El mínimo de la función está en ({x[np.argmin(f2(x,y))]},{y[np.argmin(f2(x,y))]},{np.min(f2(x,y))})')

# Parámetros para el algoritmo
num_particulas = 500
mins = [-100, -100]  # Rango de búsqueda para cada dimensión en 2D
maxs = [100, 100]

c1 = [2.5, 0.5]  # c1 dinámico (de 2.5 a 0.5)
c2 = [0.5, 2.5]  # c2 dinámico (de 0.5 a 2.5)
max_iteraciones = 100
tolerancia = 0.1 #Una tolerancia más alta permite que el algoritmo continúe explorando durante más tiempo antes de detenerse, lo que podría ayudar a escapar de mínimos locales
funcion = f2

# Ejecutar el algoritmo PSO para la función en 2D
mejor_solucion = algoritmo(num_particulas, mins, maxs, funcion, c1, c2, max_iteraciones, tolerancia)

# Evaluar el valor de la función en la mejor solución encontrada por el PSO
mejor_valor = funcion(mejor_solucion[0], mejor_solucion[1])

# Mostrar resultados
#print(f"\n### Resultados del Algoritmo PSO en 2D ###")
#print(f"Mejor solución encontrada en x = {mejor_solucion[0]:.4f}, y = {mejor_solucion[1]:.4f}")
print(f"Minimo: {mejor_valor}")
