import tensorflow as tf
import matplotlib.pyplot as plt
from src.models import create_encoder, create_decoder

def load_models(latent_dim, input_shape):
    """
    Charge les modèles d'encodeur et de décodeur sauvegardés.

    Args:
        latent_dim (int): Dimension de l'espace latent.
        input_shape (tuple): Forme des images d'entrée.

    Returns:
        encoder (tf.keras.Model): Modèle de l'encodeur.
        decoder (tf.keras.Model): Modèle du décodeur.
    """
    encoder = tf.keras.models.load_model('models/encoder.keras')
    decoder = tf.keras.models.load_model('models/decoder.keras')
    return encoder, decoder

def compare_images(encoder, decoder, sample_images, latent_dim, input_shape):
    """
    Compare les images originales et reconstruites, et affiche les résultats.

    Args:
        encoder (tf.keras.Model): Modèle de l'encodeur.
        decoder (tf.keras.Model): Modèle du décodeur.
        sample_images (numpy.ndarray): Images d'exemple à comparer.
        latent_dim (int): Dimension de l'espace latent.
        input_shape (tuple): Forme des images d'entrée.
    """
    latent_representations = encoder.predict(sample_images)
    reconstructed_images = decoder.predict(latent_representations)

    # Comparer les tailles en octets avant et après compression
    original_size = sample_images.size * sample_images.itemsize
    compressed_size = latent_representations.size * latent_representations.itemsize

    print(f"Taille originale : {original_size} octets")
    print(f"Taille après compression : {compressed_size} octets")
    compression_ratio = original_size / compressed_size
    print(f"Ratio de compression : {compression_ratio:.2f}")

    # Comparer les images originales et reconstruites
    plt.figure(figsize=(12, 6))
    for i in range(len(sample_images)):
        plt.subplot(2, len(sample_images), i + 1)
        plt.imshow(sample_images[i])
        plt.title("Originale")
        plt.axis('off')

        plt.subplot(2, len(sample_images), i + 1 + len(sample_images))
        plt.imshow(reconstructed_images[i])
        plt.title("Reconstituée")
        plt.axis('off')
    plt.savefig('models/comparison.png')
    plt.show()

def main():
    # Charger les modèles
    latent_dim = 256
    input_shape = (32, 32, 3)
    encoder, decoder = load_models(latent_dim, input_shape)

    # Charger les données de test
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0

    # Sélectionner des images d'exemple
    sample_images = x_test[5:10]

    # Comparer les images
    compare_images(encoder, decoder, sample_images, latent_dim, input_shape)

if __name__ == "__main__":
    main()
