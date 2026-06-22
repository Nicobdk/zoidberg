"""
Module de chargement et génération de données.
Centralise toutes les fonctions de data loading pour éviter duplication dans notebooks.
"""

import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


class DataConfig:
    """Configuration centralisée pour les données."""

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    # Chemins
    DATA_DIR = Path("data/raw/chest_Xray")
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"

    # Classes
    CLASS_NAMES = ['Normal', 'Bactérie', 'Virus']
    BINARY_CLASSES = ['Normal', 'Pneumonie']


def create_binary_generators(data_dir=None,
                             img_size=None,
                             batch_size=None,
                             val_split=None):
    """
    Crée les générateurs pour classification binaire (Normal vs Pneumonie).

    Utilisé par :
    - EfficientNet Binary (Stage 1 du pipeline)
    - CNN Binary from scratch

    Args:
        data_dir (str, optional): Chemin vers les données.
                                 Default: DataConfig.DATA_DIR
        img_size (tuple, optional): Taille des images (H, W).
                                   Default: (224, 224)
        batch_size (int, optional): Taille des batchs. Default: 32
        val_split (float, optional): Proportion validation. Default: 0.2

    Returns:
        tuple: (train_gen, val_gen, test_gen)

    Example:
        >>> train_gen, val_gen, test_gen = create_binary_generators()
        >>> print(f"Classes: {train_gen.class_indices}")
        >>> # Entraînement
        >>> model.fit(train_gen, validation_data=val_gen, epochs=50)
    """
    # Valeurs par défaut
    if data_dir is None:
        data_dir = DataConfig.DATA_DIR
    if img_size is None:
        img_size = DataConfig.IMG_SIZE
    if batch_size is None:
        batch_size = DataConfig.BATCH_SIZE
    if val_split is None:
        val_split = DataConfig.VALIDATION_SPLIT

    # Générateur d'augmentation pour train/val
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split
    )

    # Générateur sans augmentation pour test
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Train generator
    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    # Validation generator
    val_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    # Test generator
    test_gen = test_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False  # Important pour évaluation
    )

    return train_gen, val_gen, test_gen


def create_multiclass_generators(data_dir=None,
                                 img_size=None,
                                 batch_size=None,
                                 val_split=None):
    """
    Crée les générateurs pour classification 3-classes (Normal/Bactérie/Virus).

    Utilisé par :
    - EfficientNet Multi-class
    - CNN Multi-class from scratch

    Args:
        data_dir (str, optional): Chemin vers les données
        img_size (tuple, optional): Taille des images (H, W)
        batch_size (int, optional): Taille des batchs
        val_split (float, optional): Proportion validation

    Returns:
        tuple: (train_gen, val_gen, test_gen)

    Example:
        >>> train_gen, val_gen, test_gen = create_multiclass_generators()
        >>> print(f"Classes: {train_gen.class_indices}")
        >>> # {0: 'Normal', 1: 'Bactérie', 2: 'Virus'}
    """
    # Valeurs par défaut
    if data_dir is None:
        data_dir = DataConfig.DATA_DIR
    if img_size is None:
        img_size = DataConfig.IMG_SIZE
    if batch_size is None:
        batch_size = DataConfig.BATCH_SIZE
    if val_split is None:
        val_split = DataConfig.VALIDATION_SPLIT

    # Générateur d'augmentation pour train/val
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split
    )

    # Générateur sans augmentation pour test
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Train generator
    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",  # 3 classes
        subset="training"
    )

    # Validation generator
    val_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    # Test generator
    test_gen = test_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def create_augmented_generator(data_dir=None,
                               img_size=None,
                               batch_size=None,
                               val_split=None,
                               rotation_range=20,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               brightness_range=(0.8, 1.2)):
    """
    Crée des générateurs avec data augmentation agressive.

    Utilisé pour :
    - Améliorer robustesse
    - Compenser déséquilibre classes

    Args:
        data_dir (str, optional): Chemin vers les données
        img_size (tuple, optional): Taille des images
        batch_size (int, optional): Taille des batchs
        val_split (float, optional): Proportion validation
        rotation_range (int): Rotation max en degrés
        zoom_range (float): Zoom max (0.2 = ±20%)
        horizontal_flip (bool): Flip horizontal aléatoire
        brightness_range (tuple): Range de luminosité

    Returns:
        tuple: (train_gen, val_gen, test_gen)

    Example:
        >>> train_gen, val_gen, test_gen = create_augmented_generator(
        ...     rotation_range=30,
        ...     zoom_range=0.3
        ... )
    """
    # Valeurs par défaut
    if data_dir is None:
        data_dir = DataConfig.DATA_DIR
    if img_size is None:
        img_size = DataConfig.IMG_SIZE
    if batch_size is None:
        batch_size = DataConfig.BATCH_SIZE
    if val_split is None:
        val_split = DataConfig.VALIDATION_SPLIT

    # Générateur avec augmentation agressive
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        brightness_range=brightness_range,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        validation_split=val_split
    )

    # Générateur sans augmentation pour val/test
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Train generator (avec augmentation)
    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    # Validation generator (sans augmentation)
    val_gen = test_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    # Test generator
    test_gen = test_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen
