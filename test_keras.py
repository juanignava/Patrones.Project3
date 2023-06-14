import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from PIL import Image

# Directorio que contiene las imágenes
image_directory = "images/analysis_dataset/analysis_dataset/"

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
        porcentaje = 2
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

# Aplanar las matrices de imágenes a un formato bidimensional
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Convertir las imágenes a tensores
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

# Entrenar el modelo K-Means para generar etiquetas de clúster
kmeans = KMeans(n_clusters=38)
kmeans.fit(X_train)

# Obtener las etiquetas de clúster asignadas a los puntos de datos
labels_train = kmeans.labels_
print(labels_train)
labels_test = kmeans.predict(X_test)
print(labels_test) 

# Definir la función de pérdida para el clustering K-Means
def kmeans_loss(y_true, y_pred):
    if y_true.dtype != tf.float32:
        y_true = tf.cast(y_true, tf.float32)
    if y_pred.dtype != tf.float32:
        y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Definir el modelo K-Means
input_shape = X_train.shape[1:]
model_input = layers.Input(shape=input_shape)
kmeans_output = layers.Dense(units=38, activation='softmax')(model_input)

# Compilar el modelo
model = Model(inputs=model_input, outputs=kmeans_output)
model.compile(optimizer='adam', loss=kmeans_loss)

# Entrenar el modelo utilizando las etiquetas de clúster generadas
model.fit(X_train, labels_train, epochs=10, batch_size=batch_size)

# Obtener las etiquetas de clúster asignadas a los puntos de datos
labels_pred_train = model.predict(X_train)
labels_pred_test = model.predict(X_test)

# Imprimir las etiquetas de clúster
print(labels_pred_train)


