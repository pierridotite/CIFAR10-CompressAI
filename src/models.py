from tensorflow.keras import layers, models

def create_encoder(input_shape, latent_dim):
    """
    Crée l'architecture de l'encodeur.

    Args:
        input_shape (tuple): Forme des images d'entrée.
        latent_dim (int): Dimension de l'espace latent.

    Returns:
        model (tf.keras.Model): Modèle de l'encodeur.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(latent_dim, activation='tanh')
    ])
    return model

def create_decoder(latent_dim, output_shape):
    """
    Crée l'architecture du décodeur.

    Args:
        latent_dim (int): Dimension de l'espace latent.
        output_shape (tuple): Forme des images de sortie.

    Returns:
        model (tf.keras.Model): Modèle du décodeur.
    """
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(4 * 4 * 128, activation='relu'),
        layers.Reshape((4, 4, 128)),
        layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return model
