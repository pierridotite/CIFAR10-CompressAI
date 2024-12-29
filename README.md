# CIFAR10-CompressAI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.12.0-orange.svg)

## Description

**CIFAR10-CompressAI** est un projet implémentant un autoencodeur convolutionnel pour la compression et la reconstruction d'images du jeu de données CIFAR-10. L'autoencodeur est entraîné en utilisant une combinaison de perte perceptuelle et de perte MSE, offrant une compression efficace tout en préservant la qualité des images reconstruites.

![Courbe de Loss](models/loss_curves.png)
![Comparaison de Compression](models/comparison.png)

## Fonctionnalités

- **Compression efficace** : Utilisation d'un autoencodeur convolutionnel pour réduire la taille des images CIFAR-10.
- **Reconstructions de haute qualité** : Combinaison de pertes perceptuelle et MSE pour maintenir la qualité des images reconstruites.
- **Data Augmentation** : Techniques avancées pour améliorer la robustesse du modèle.
- **Support GPU** : Optimisé pour l'entraînement sur GPU avec TensorFlow.
- **Modularité** : Code organisé en modules pour faciliter les contributions et les extensions.

## Structure du Projet

```
CIFAR10-CompressAI/
├── data/                  # Dossier pour les données
├── models/                # Sauvegarde des modèles entraînés et des images
├── notebooks/             # Jupyter notebooks pour l'exploration
├── src/                   # Code source
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
├── .gitignore             # Fichiers et dossiers à ignorer par Git
├── README.md              # Documentation du projet
├── requirements.txt       # Dépendances du projet
├── LICENSE                # Licence du projet
├── CONTRIBUTING.md        # Guide de contribution
```

## Installation

1. **Cloner le dépôt**

    ```bash
    git clone https://github.com/pierridotite/CIFAR10-CompressAI.git
    cd CIFAR10-CompressAI
    ```

2. **Créer un environnement virtuel (optionnel mais recommandé)**

    ```bash
    python -m venv venv
    venv\Scripts\activate      # Sur Windows
    source venv/bin/activate   # Sur macOS/Linux
    ```

3. **Installer les dépendances**

    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### Entraînement du Modèle

Pour entraîner l'autoencodeur, exécutez :

```bash
python src/train.py
```

### Évaluation du Modèle

Pour évaluer et comparer les images originales et reconstruites, exécutez :

```bash
python src/evaluate.py
```

## Contribution

Les contributions sont les bienvenues ! Veuillez consulter le fichier [CONTRIBUTING.md](CONTRIBUTING.md) pour plus de détails.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Exemples

### Entraînement

![Exemple d'entraînement](models/loss_curves.png)

### Comparaison de Compression

![Exemple de compression](models/comparison.png)

## Utilisation Avancée

Vous pouvez explorer les notebooks dans le dossier `notebooks/` pour des analyses et visualisations supplémentaires.

---

Merci d'utiliser **CIFAR10-CompressAI** ! N'hésitez pas à contribuer et à partager ce projet avec la communauté.