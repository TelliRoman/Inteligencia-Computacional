import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Cargar el dataset Iris
df = pd.read_csv('Guia4/irisbin_trn.csv')
arreglo = df.to_numpy()
x = arreglo[: , :-3] 


k_values = list(range(2, 11)) #Probamos k con valores de 2 a 10
inertia_values = [] #inertia_values es donde almacenaremos los valores de inercia (una medida de qué tan compactos son los clusters).
silhouette_scores = [] #silhouette_scores almacenará los puntajes de la silueta, que mide la separación de los clusters.

#La inercia mide la suma de las distancias cuadradas de los puntos a su centroide más cercano. 
# Cuanto más baja la inercia, mejor están ajustados los puntos dentro de sus clusters.

#El puntaje de siulueta Es una métrica que evalúa cuán bien están separados los clusters.
#Va de -1 a 1, donde 1 significa que los clusters están muy bien separados y -1 que los puntos están mal agrupados.

#Un cluster (o clúster en español) es un grupo de datos o puntos en un conjunto de datos que tienen características similares entre sí

# Probar diferentes valores de k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x) #se entrenan los clusters.
    inertia_values.append(kmeans.inertia_)
    
    # Calcular el puntaje de la métrica de silueta
    silhouette_avg = silhouette_score(x, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Graficar los resultados
plt.figure(figsize=(10, 5))

# Gráfico de inercia
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Inercia vs número de clusters')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')

# Gráfico de puntaje de silueta
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', color='red')
plt.title('Puntaje de silueta vs número de clusters')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Puntaje de silueta')

plt.tight_layout()
plt.show()
# Imprimir el mejor valor de k según el puntaje de silueta
best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f"El valor óptimo de k es: {best_k}")
