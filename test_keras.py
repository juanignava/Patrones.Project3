import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

# Directorio que contiene las imágenes
image_directory = "images/analysis_dataset/"

# Inicializar una lista vacía para almacenar las imágenes
images = []
images_test = []

# Tamaño del lote
batch_size = 100

# Recorrer las subcarpetas dentro del directorio
for root, dirs, files in os.walk(image_directory):
    for directory in dirs:
        subdir = os.path.join(root, directory)
        # Obtener la lista de nombres de archivo de las imágenes en la subcarpeta
        image_files = os.listdir(subdir)
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
                images_test.append(batch_X_test)

# Concatenar los lotes en un solo array
X_train = np.concatenate(images)
X_test = np.concatenate(images_test)

# Definir el modelo K-Means
input_shape = X_train.shape[1:]
model_input = layers.Input(shape=input_shape)
kmeans = layers.KMeans(num_clusters=38)(model_input)

# Compilar el modelo
model = tf.keras.Model(inputs=model_input, outputs=kmeans)
model.compile(optimizer='adam', loss='kld')

# Entrenar el modelo
model.fit(X_train, X_train, epochs=10)

# Obtener las etiquetas de clúster asignadas a los puntos de datos
labels = model.predict(X_train)

# Imprimir las etiquetas de clúster
print(labels)
