import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Directorio que contiene las imágenes
image_directory = "images/analysis_dataset"

# Inicializar una lista vacía para almacenar las imágenes
images = []
images_test = []

# Tamaño del lote
batch_size = 5

# Recorrer las subcarpetas dentro del directorio
for root, dirs, files in os.walk(image_directory):
    for directory in dirs:
        subdir = os.path.join(root, directory)
        # Obtener la lista de nombres de archivo de las imágenes en la subcarpeta
        image_files = os.listdir(subdir)
        #print(len(image_files))
        # Procesar las imágenes en lotes
        total = len(image_files)
        porcentaje = 80
        entrenamiento = int((total/100)*porcentaje)
        prueba = int(total - entrenamiento)
        for i in range(0, len(image_files), batch_size):
            # Cargar y convertir las imágenes en matrices numpy para entrenamiento
            if (i <= entrenamiento):
                batch_images = []
                for file in image_files[i:i+batch_size]:
                    image_path = os.path.join(subdir, file)
                    image = Image.open(image_path)
                    image_array = np.array(image)
                    batch_images.append(image_array)
                
                # Concatenar las matrices del lote en un solo array
                batch_X = np.concatenate(batch_images)
                images.append(batch_X)
            else:
                # Cargar y convertir las imágenes en matrices numpy para pruebas
                batch_images_test = []
                for file in image_files[i:i+batch_size]:
                    image_path = os.path.join(subdir, file)
                    image = Image.open(image_path)
                    image_array = np.array(image)
                    batch_images_test.append(image_array)
                
                # Concatenar las matrices del lote en un solo array
                batch_X_test = np.concatenate(batch_images_test)
                images_test.append(batch_X)

# Concatenar los lotes en un solo array
X_train = np.concatenate(images)
X_test = np.concatenate(images_test)

# Aplanar X
X_train = X_train.reshape(-1, 3)  # Aplanar las imágenes en un vector unidimensional (num_imagenes * 10000, 3)
print(len(X_train))

X_test = X_test.reshape(-1, 3)  # Aplanar las imágenes en un vector unidimensional (num_imagenes * 10000, 3)
print(len(X_test))

X_train = X_train.astype(np.float16) / 255.0
X_test = X_test.astype(np.float16) / 255.0

print(X_train[0])
print(X_test[0])

def k_means_test(X_train):
    kmeans = KMeans(n_clusters=38, max_iter=2)
    kmeans.fit(X_train)

    # Obtener las etiquetas de cluster asignadas a cada punto de datos
    labels = kmeans.labels_

    # Obtener las coordenadas de los centroides
    centroids = kmeans.cluster_centers_

    # Imprimir las etiquetas de cluster y los centroides
    print("Etiquetas de cluster:", labels)
    print("Coordenadas de los centroides:", centroids)

def spectral_test(X):

    # Utilizar Spectral Clustering
    spectral_clustering = SpectralClustering(n_clusters=38, affinity='nearest_neighbors', n_init=2)
    spectral_clustering.fit(X)

    # Obtener las etiquetas de cluster asignadas a cada punto de datos
    labels = spectral_clustering.labels_

    # Imprimir las etiquetas de cluster
    print("Etiquetas de cluster:", labels)

k_means_test(X_train)
#spectral_test(X)