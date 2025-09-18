import numpy as np
import pandas as pd

#PASO 1: Inicializar feromonas con la forma de la matriz de las ciudades
def inicializar_feromonas(num_ciudades, sigma_0):
    feromonas = np.full((num_ciudades, num_ciudades), sigma_0)
    return feromonas

#PASO 2: Ubicar N hormigas en el nodo de origen, TODAS LAS HORMIGAS PARTEN DEL MISMO NODO
def inicializar_hormigas(n_hormigas, nodo_origen):
    # Todas las hormigas comienzan en el nodo de origen
    posiciones_hormigas = np.full(n_hormigas, nodo_origen)
    return posiciones_hormigas

# Función para calcular la probabilidad de elegir una ciudad
def seleccionar_ciudad(matriz_distancias,feromonas,ciudad_actual,ciudades_visitadas,alpha=1.0, beta=5.0):
    num_ciudades = matriz_distancias.shape[0]
    probabilidades = np.zeros(num_ciudades)

    # Lista de ciudades que aún no han sido visitadas (Nodos vecinos sin tabú)
    nodos_no_visitados = []
    for j in range(num_ciudades):
        if j not in ciudades_visitadas and j!= ciudad_actual:
            nodos_no_visitados.append(j) #agreamos la ciudad 
        
    # Calculamos la probabilidad de ir a cada ciudad no visitada
    for ciudad in nodos_no_visitados:
        tau_ij = feromonas[ciudad_actual, ciudad] ** alpha
        eta_ij = (1.0 / matriz_distancias[ciudad_actual, ciudad]) ** beta
        probabilidades[ciudad] = tau_ij * eta_ij

    suma_probabilidades = np.sum(probabilidades)
    if suma_probabilidades > 0:
        probabilidades /= suma_probabilidades
        # Elegimos la siguiente ciudad basada en las probabilidades
        siguiente_ciudad = np.random.choice(range(num_ciudades), p=probabilidades)
        # Evitamos ciudades que ya hayan sido visitadas
        while siguiente_ciudad in ciudades_visitadas:
            siguiente_ciudad = np.random.choice(range(num_ciudades), p=probabilidades)
        return siguiente_ciudad
    return None

#Calculamos la longitud de un camino
def calcular_longitud_camino(camino, matriz_distancias):
    longitud = 0
    for i in range(len(camino) - 1): #hasta - 1 porque vamos sumando de a pares
        longitud += matriz_distancias[camino[i], camino[i + 1]]
    return longitud

#########################################
#EVAPORACION Y DEPOSITO DE FEROMONAS#####
##########################################

#Cuanto más corto es el camino, mayor es el valor que se deposita de feromona.
def actualizar_feromonas_global(feromonas, mejor_camino, matriz_distancias, rho=0.1, theta=1.0):
    # Evaporar feromonas en todas las conexiones
    feromonas *= (1 - rho)

    # Calcular la longitud del mejor camino
    longitud_mejor_camino = calcular_longitud_camino(mejor_camino, matriz_distancias)

    # Añadir feromonas solo al mejor camino
    for i in range(len(mejor_camino) - 1):
        ciudad_actual = mejor_camino[i]
        siguiente_ciudad = mejor_camino[i + 1]
        feromonas[ciudad_actual, siguiente_ciudad] += theta / longitud_mejor_camino
        feromonas[siguiente_ciudad, ciudad_actual] += theta / longitud_mejor_camino

    return feromonas

def actualizar_feromonas_uniforme(feromonas, caminos, matriz_distancias, rho=0.1, theta=1.0):
    # Evaporar feromonas en todas las conexiones
    feromonas *= (1 - rho)

    # Añadir feromonas de manera uniforme en todos los caminos
    for camino in caminos:
        longitud_camino = calcular_longitud_camino(camino, matriz_distancias)
        for i in range(len(camino) - 1):
            ciudad_actual = camino[i]
            siguiente_ciudad = camino[i + 1]
            feromonas[ciudad_actual, siguiente_ciudad] += theta 
            feromonas[siguiente_ciudad, ciudad_actual] += theta

    return feromonas

def actualizar_feromonas_local(feromonas, caminos, matriz_distancias, rho=0.1, theta=1.0):
    # Evaporar las feromonas (localmente) después de cada recorrido de hormigas
    feromonas *= (1 - rho)

    #Añadir feromonas solo en las rutas recorridas por cada hormiga
    for camino in caminos:
        longitud_camino = calcular_longitud_camino(camino,matriz_distancias)
        for i in range(len(camino)-1):
            ciudad_actual = camino[i]
            siguiente_ciudad = camino[i + 1]
            feromonas[ciudad_actual, siguiente_ciudad] += theta / longitud_camino
            feromonas[siguiente_ciudad, ciudad_actual] += theta / longitud_camino
     
    return feromonas


#PASO 3: Inicial el algoritmo
def algoritmo(matriz_distancias, feromonas, posiciones_hormigas, max_iteraciones,tipo_actualizacion, alpha=1.0, beta=5.0):
    num_ciudades = matriz_distancias.shape[0]
    caminos = [[] for _ in range(n_hormigas)] # Crear una lista vacía de caminos para cada hormiga
    mejor_camino_global = None # Inicialmente no hay un mejor camino global
    mejor_longitud_global = float('inf')
    iteraciones_sin_cambio = 0 

    for iteracion in range(max_iteraciones):
       
        print(f"\nIteración: {iteracion + 1}")
        caminos_anterior_iteracion = caminos.copy()  # Guardamos los caminos de la iteración anterior
        longitudes = []
       
        for i in range(n_hormigas):
            ciudades_visitadas = [posiciones_hormigas[i]]  # La hormiga comienza en su posición inicial

            #Mientras no hayamos visitado todas las ciudades
            while(len(ciudades_visitadas) < matriz_distancias.shape[0]):
                ciudad_actual = ciudades_visitadas[-1]
                siguiente_ciudad = seleccionar_ciudad(matriz_distancias, feromonas, ciudad_actual, ciudades_visitadas, alpha, beta) #selecciona la siguiente ciudad mediante la función seleccionar_ciudad, que toma en cuenta las feromonas y las distancias.
                ciudades_visitadas.append(siguiente_ciudad) #una vez seleccionada añadimos a la lista
                        
            ciudades_visitadas.append(posiciones_hormigas[i]) # Regresa a la ciudad inicial
            caminos[i] = ciudades_visitadas
            longitud_camino = calcular_longitud_camino(ciudades_visitadas, matriz_distancias)
            longitudes.append(longitud_camino)
            print(f"Hormiga {i + 1}: Longitud: {longitud_camino}")     

        # Actualización de feromonas según el tipo seleccionado
        if tipo_actualizacion == 'global':
            if mejor_camino_global is not None: #El mejor camino global no esta disponible en la 1er iteracion
                actualizar_feromonas_global(feromonas, mejor_camino_global, matriz_distancias, rho=0.1, theta=1.0)
        elif tipo_actualizacion == 'uniforme':
            actualizar_feromonas_uniforme(feromonas, caminos, matriz_distancias, rho=0.1, theta=1.0)
        elif tipo_actualizacion == 'local':
            actualizar_feromonas_local(feromonas, caminos, matriz_distancias, rho=0.1, theta=1.0)

        # Verificar si todas las hormigas siguen el mismo camino que en la iteración anterior
        if caminos == caminos_anterior_iteracion:
            iteraciones_sin_cambio += 1
        else:
            iteraciones_sin_cambio = 0  # Reiniciar contador si los caminos cambian
        
        # Criterio de corte: Si todas las hormigas siguen el mismo camino durante 5 iteraciones consecutivas
        if iteraciones_sin_cambio >= 5:
            print(f"Criterio de corte alcanzado tras {iteracion + 1} iteraciones: todas las hormigas siguen el mismo camino durante 5 iteraciones consecutivas.")
            break

        # Encontrar el mejor camino de la iteración
        mejor_camino_iteracion = caminos[np.argmin(longitudes)]
        mejor_longitud_iteracion = min(longitudes)

        # Comparar con el mejor camino global
        if mejor_longitud_iteracion < mejor_longitud_global:
            mejor_camino_global = mejor_camino_iteracion
            mejor_longitud_global = mejor_longitud_iteracion

    #    print(f"Mejor camino global hasta ahora: {mejor_camino_global}, Longitud: {mejor_longitud_global}")
    #    print(f"Matriz de feromonas después de la iteración {iteracion + 1}:\n{feromonas}")

    # Convertir cada camino en la lista de caminos a enteros estándar de Python
    caminos = [[int(ciudad) for ciudad in camino] for camino in caminos]

    # Convertir el mejor camino global a enteros estándar de Python antes de retornarlo
    mejor_camino_global = [int(ciudad) for ciudad in mejor_camino_global]

    return caminos,mejor_camino_global, int(mejor_longitud_global)
           


# Cargar la matriz de distancias desde el archivo CSV
df = pd.read_csv("gr17.csv",header=None)
matriz_distancias = df.to_numpy()

#print(matriz_distancias)
columnas = matriz_distancias.shape[0]
filas = matriz_distancias.shape[1]

#PASO 1 Inicializar feromonas con la forma de la matriz de las ciudades:
feromonas = inicializar_feromonas(filas,sigma_0=0.1)
#print(feromonas)

#PASO 2: Ubicar N hormigas en el nodo de origen, TODAS LAS HORMIGAS PARTEN DEL MISMO NODO
n_hormigas = 20  # Número de hormigas
nodo_origen = 0  # Nodo de origen (puede ser el nodo 0 o cualquier otro nodo)
posiciones_hormigas = inicializar_hormigas(n_hormigas, nodo_origen)
#print(posiciones_hormigas)

# PASO 3: Ejecutar el algoritmo
max_iteraciones = 1500
tipo_actualizacion="global"
caminos,mejor_camino_global, mejor_longitud_global = algoritmo(matriz_distancias, feromonas, posiciones_hormigas, max_iteraciones,tipo_actualizacion, alpha=1.0, beta=5.0)
print(f'caminos de cada hormiga: {caminos}')
print(f'mejor camino global: {mejor_camino_global} con longitud: {mejor_longitud_global}')

#LOCAL Y UNIFORME NO CONVERGEN Y NO ENCUENTRAN EL OPTIMO, ESTA BIEN O MAL???????? A CHEQUEAR

#para la tabla repetir 10 veces y sacar el tiempo promedio, la distancia promedio y el numero de iteraciones promedio

#FEROMONAS:
#GLOBAL: Dejar una fraccion de ese total de feromonas que depende de la longitud total del camino que tenemos
#UNIFORME: Constante de feromonas en cada una de las transiciones
#LOCAL: Cantidad de feromonas / distancia que hay sobre ciudad i a la ciudad j

#Diferencias clave:
#En la actualización global, las feromonas se añaden solo al mejor camino.
#En la uniforme, se añaden a todos los caminos recorridos de manera uniforme.
#En la local, las feromonas se añaden inmediatamente después de cada recorrido, pero proporcionalmente a la longitud de cada camino.


#La matriz de feromonas es de la misma estructura que la matriz de caminos


#Lista de nodos tabu: sacar el nodo del que venimos B-> A de A no podemos volver a B

#Luego de alcanzar la lista por ej: [A,B,C,D] ya se alcanzo al final y debemos agregar a mano [A,B,C,D,A] el nodo origen
#y calculamos la longitud sumando las distancias

#Para el criterio d corte es hasta que todas las hormigas sigan el mismo camino
#DEBEMOS TENER EN CUENTA:  SI TODAS LAS HORMIGAS PARTIERON DEL MISMO NODO -> COMPARAMOS LA SECUENCIA
                        #  SINO -> COMPARAMOS LOS ELEMENTOS DEL CAMINO

#Para el criterio d corte es hasta que todas las hormigas sigan el mismo camino
#Tener un contador, si todas las hormigas se alinean durante 5 iteraciones consecutivas ahi si detenemos

