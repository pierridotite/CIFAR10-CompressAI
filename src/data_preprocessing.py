import tensorflow as tf
import numpy as np

def load_and_preprocess_data():
    """
    Charge le jeu de données CIFAR-10 et normalise les images.

    Returns:
        x_train (numpy.ndarray): Données d'entraînement normalisées.
        x_test (numpy.ndarray): Données de test normalisées.
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, x_test
