# Patrones.Project3
Unsupervised learning models

This project compares machine learning models to clasify illness in plants by analysing pictures of their leaves.

## Data set

The dataset used to train and test these models is the one located at: https://www.tensorflow.org/datasets/catalog/plant_village?hl=es-419

In the Jupyter notebook `unsupervised_learning.ipynb` you can find two experiments that try to demostrate two hypotesis.

1. A dataset without labels could create an autoencoder and with this autoencoder generate a model that gives better results.

2. Adding noise into the autoencoder input posibly creates a better clasification.