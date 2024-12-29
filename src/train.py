import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data_preprocessing import load_and_preprocess_data
from src.models import create_encoder, create_decoder
import matplotlib.pyplot as plt
import numpy as np

def check_and_filter_data(data):
    """
    Vérifie et filtre les données pour les valeurs None et les images invalides.

    Args:
        data (numpy.ndarray): Données d'images.

    Returns:
        numpy.ndarray: Données filtrées.
    """
    filtered_data = []
    for img in data:
        if img is not None and img.size > 0 and not np.isnan(img).any():
            filtered_data.append(img)
        else:
            print("Image invalide détectée et filtrée.")
    return np.array(filtered_data)

def main():
    # Chargement et prétraitement des données
    x_train, x_test = load_and_preprocess_data()
    x_train = check_and_filter_data(x_train)
    x_test = check_and_filter_data(x_test)

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Après le filtrage, les données d'entraînement ou de test sont vides. Vérifiez vos données.")

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(256).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
    val_dataset = val_dataset.batch(256).prefetch(tf.data.AUTOTUNE)

    input_shape = (32, 32, 3)
    latent_dim = 256

    # Création des modèles
    encoder = create_encoder(input_shape=input_shape, latent_dim=latent_dim)
    decoder = create_decoder(latent_dim=latent_dim, output_shape=input_shape)

    # Construction de l'autoencodeur
    autoencoder_input = tf.keras.Input(shape=input_shape)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = tf.keras.Model(inputs=autoencoder_input, outputs=decoded)

    # Compilation de l'autoencodeur
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    # Callbacks
    callbacks_autoencoder = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('models/best_autoencoder.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    # Entraînement
    history_autoencoder = autoencoder.fit(
        train_dataset,
        steps_per_epoch=max(len(x_train) // 256, 1),
        validation_data=val_dataset,
        epochs=100,
        callbacks=callbacks_autoencoder
    )

    # Sauvegarder les modèles
    encoder.save('models/encoder.keras')
    decoder.save('models/decoder.keras')

    # Sauvegarder l'historique des pertes
    plt.figure(figsize=(10, 5))
    plt.plot(history_autoencoder.history['loss'], label='Perte Entraînement')
    plt.plot(history_autoencoder.history['val_loss'], label='Perte Validation')
    plt.title('Courbes de Perte - Autoencodeur')
    plt.xlabel('Époques')
    plt.ylabel('Perte (MSE)')
    plt.legend()
    plt.grid()
    plt.savefig('models/loss_curves.png')

if __name__ == "__main__":
    main()
