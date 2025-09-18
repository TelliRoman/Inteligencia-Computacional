import numpy as np
import time

# Función para inicializar el enjambre aleatorio con velocidades iniciales nulas
def enjambre(num_particulas, mins, maxs):
    posiciones = np.random.uniform(mins, maxs, (num_particulas, len(mins)))  # Posiciones aleatorias en el rango
    velocidades = np.zeros((num_particulas, len(mins)))  # Velocidades iniciales nulas
    mejor_pos_personal = np.copy(posiciones)
    return posiciones, velocidades, mejor_pos_personal

def algoritmo(num_particulas, mins, maxs, funcion, c1, c2, max_iteraciones, tolerancia):
    
    inicio = time.perf_counter()    # Inicio del contador de tiempo para ver velocidad de convergencia
    # Inicializamos el enjambre
    posiciones, velocidades, mejor_pos_personal = enjambre(num_particulas, mins, maxs)
  
    # Evaluamos la función de error en las posiciones iniciales
    valores_error_personal = funcion(posiciones)
    
    # Inicializamos la mejor posición global
    mejor_pos_global = mejor_pos_personal[np.argmin(valores_error_personal)]
    mejor_error_global = np.min(valores_error_personal)

    c1_inicial, c1_final = c1
    c2_inicial, c2_final = c2

    for iteracion in range(max_iteraciones):
        # Imprimir el número de la iteración
        print(f"Iteración {iteracion + 1}")
        # Decrementamos c1 e incrementamos c2 por cada iteración
        c1_actual = c1_inicial - (c1_inicial - c1_final) * (iteracion / max_iteraciones)
        c2_actual = c2_inicial + (c2_final - c2_inicial) * (iteracion / max_iteraciones)

        mejor_error_anterior = mejor_error_global

        for i in range(num_particulas):
            r1 = np.random.rand()  # Componente aleatoria para el peso personal
            r2 = np.random.rand()  # Componente aleatoria para el peso global
            
            # Actualización de la velocidad (ahora para múltiples dimensiones)
            velocidades[i] = (velocidades[i] + 
                              c1_actual * r1 * (mejor_pos_personal[i] - posiciones[i]) +  # Influencia personal
                              c2_actual * r2 * (mejor_pos_global - posiciones[i]))  # Influencia global
          
            posiciones[i] = posiciones[i] + velocidades[i]
    
            posiciones[i] = np.clip(posiciones[i], mins, maxs)  # Mantener partículas dentro de los rangos permitidos
       
        valores_error_actuales = funcion(posiciones)  # Evaluamos las nuevas posiciones

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
            print(f"Convergencia alcanzada en la iteración {iteracion + 1}.")
            break
    
    fin = time.perf_counter()
    print(f'El algoritmo terminó en {fin-inicio} segundos.')
    return mejor_pos_global

#############################
###### EJEMPLO DE USO ########
#############################

# Función de error para 1D
def f1(x):
    return -x * np.sin(np.sqrt(np.abs(x)))
x = np.linspace(-512,512,100)
y_f = f1(x)
#print(f'El mínimo de la función está en: [{x[np.argmin(y_f)]}, {np.min(y_f)}]')


num_particulas = 30
mins = [-512]  # Rango de búsqueda para cada dimensión
maxs = [512]

c1 = [2.5, 0.5]  # c1 dinámico (de 2.5 a 0.5)
c2 = [0.5, 2.5]  # c2 dinámico (de 0.5 a 2.5)
max_iteraciones = 100
tolerancia = 1e-6
funcion = f1

# Ejecutar el algoritmo PSO para la función en 2D
mejor_solucion = algoritmo(num_particulas, mins, maxs, funcion, c1, c2, max_iteraciones, tolerancia)

# Evaluar el valor de la función en la mejor solución encontrada por el PSO
mejor_valor = funcion(mejor_solucion)

# Mostrar resultados
#print(f"\n### Resultados del Algoritmo PSO ###")
print(f"Mejor solución encontrada en x = {mejor_solucion[0]:.4f}")
print(f"Valor de la función en la mejor solución: {mejor_valor[0]:.4f}")