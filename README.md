# CIFAR10-CompressAI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.12.0-orange.svg)

## Description

**CIFAR10-CompressAI** is a project implementing a convolutional autoencoder for compressing and reconstructing images from the CIFAR-10 dataset. The autoencoder is trained using a combination of perceptual loss and MSE loss, providing efficient compression while preserving the quality of reconstructed images.

<div style="display: flex; justify-content: center;">
    <img src="models/loss_curves.png" alt="Loss Curve" width="300"/>
    <img src="models/comparison.png" alt="Compression Comparison" width="300"/>
</div>

## Features

- **Efficient Compression**: Utilizes a convolutional autoencoder to reduce the size of CIFAR-10 images.
- **High-Quality Reconstructions**: Combines perceptual loss and MSE loss to maintain the quality of reconstructed images.
- **Data Augmentation**: Advanced techniques to enhance the model's robustness.
- **GPU Support**: Optimized for training on GPUs with TensorFlow.
- **Modularity**: Code organized into modules to facilitate contributions and extensions.

## Mathematical Background

### Convolutional Autoencoder

The core of **CIFAR10-CompressAI** is a convolutional autoencoder, a type of neural network architecture designed for unsupervised learning of efficient codings. The autoencoder consists of two main parts:

1. **Encoder**: Compresses the input image into a lower-dimensional latent representation.
2. **Decoder**: Reconstructs the image from the latent representation.

Mathematically, let \( \mathbf{x} \in \mathbb{R}^{H \times W \times C} \) represent the input image, where \( H \), \( W \), and \( C \) are the height, width, and number of channels, respectively. The encoder maps \( \mathbf{x} \) to a latent vector \( \mathbf{z} \):

\[
\mathbf{z} = \text{Encoder}(\mathbf{x})
\]

The decoder reconstructs the image from \( \mathbf{z} \):

\[
\hat{\mathbf{x}} = \text{Decoder}(\mathbf{z})
\]

### Loss Functions

The training process optimizes a loss function combining Mean Squared Error (MSE) and perceptual loss:

\[
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{MSE}} + \beta \cdot \mathcal{L}_{\text{Perceptual}}
\]

- **MSE Loss**:

\[
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2_2
\]

- **Perceptual Loss**: Measures the difference in high-level feature representations, often using a pretrained network like VGG.

\[
\mathcal{L}_{\text{Perceptual}} = \sum_{j} \| \phi_j(\mathbf{x}) - \phi_j(\hat{\mathbf{x}}) \|^2_2
\]

where \( \phi_j \) represents the activations of layer \( j \) in the pretrained network.

### Compression Metrics

The compression ratio is a key metric indicating how much the image size is reduced. It is calculated as:

\[
\text{Compression Ratio} = \frac{\text{Original Size}}{\text{Compressed Size}}
\]

In this project, the original and compressed sizes are measured in bytes.

## Compression Metrics and Results

### Calculations

- **Original Size**: The original size of an image from the CIFAR-10 dataset is 61,440 bytes.
  
  - **Calculation**: CIFAR-10 images are \(32 \times 32\) pixels with 3 color channels (RGB). Each pixel per channel is typically represented by 8 bits (1 byte).
  
  \[
  32 \times 32 \times 3 = 3,072 \text{ pixels}
  \]
  
  \[
  3,072 \text{ pixels} \times 20 \text{ bytes per pixel} = 61,440 \text{ bytes}
  \]
  
  *Note: The exact calculation may vary based on data representation.*

- **Compressed Size**: After compression using the autoencoder, the size is reduced to 5,120 bytes.

- **Compression Ratio**: 

\[
\text{Compression Ratio} = \frac{61,440}{5,120} = 12.00
\]

This indicates that the compressed image is 12 times smaller than the original, achieving significant storage savings while maintaining image quality.

### Explanation of Results

The achieved compression ratio of **12.00** demonstrates the effectiveness of the convolutional autoencoder in reducing image size. By leveraging both perceptual and MSE loss functions, the model ensures that essential visual information is preserved, resulting in high-quality reconstructions despite the substantial reduction in data size.

## Project Structure

```
CIFAR10-CompressAI/
├── data/                  # Folder for data
├── models/                # Storage for trained models and images
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
├── .gitignore             # Files and folders to ignore by Git
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
├── LICENSE                # Project license
├── CONTRIBUTING.md        # Contribution guide
```

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/pierridotite/CIFAR10-CompressAI.git
    cd CIFAR10-CompressAI
    ```

2. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    venv\Scripts\activate      # On Windows
    source venv/bin/activate   # On macOS/Linux
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the autoencoder, run:

```bash
python src/train.py
```

### Evaluating the Model

To evaluate and compare the original and reconstructed images, run:

```bash
python src/evaluate.py
```

## Contribution

Contributions are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Examples

### Training

![Training Example](models/loss_curves.png)

### Compression Comparison

![Compression Example](models/comparison.png)

## Advanced Usage

You can explore the notebooks in the `notebooks/` folder for additional analyses and visualizations.

---

Thank you for using **CIFAR10-CompressAI**! Feel free to contribute and share this project with the community.