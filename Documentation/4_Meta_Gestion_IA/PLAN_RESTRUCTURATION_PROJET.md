# 🏗️ PLAN DE RESTRUCTURATION - PROJET ZOIDBERG
## Architecture Professionnelle Deep Learning

**Objectif** : Transformer le projet en structure professionnelle avec code modulaire  
**Principe** : Notebooks = démonstration / src = code réutilisable  
**Durée estimée** : 3-4 heures

---

# 🎯 PROBLÈME ACTUEL

## Ce qu'on a maintenant
```
zoidberg/
├── notebooks/           # ❌ Code dupliqué dans chaque notebook
│   ├── experiments_v3.ipynb           (200 KB)
│   ├── experiments_v3_FIXED.ipynb     (20 KB)
│   ├── 07_comprehensive_analysis.ipynb (800 KB)
│   └── ... (8 notebooks)
│
├── src/                 # ⚠️  Presque vide
│   ├── visualization/
│   │   ├── data_exploration.py
│   │   └── grad_cam.py
│   └── features/
│
├── scripts/             # ❌ Scripts standalone, pas réutilisables
│   ├── generate_comparative_report.py
│   ├── generate_flow_diagram.py
│   └── ...
│
└── models/
    ├── trained/         # ✅ OK
    └── evaluation/      # ✅ OK
```

## Problèmes identifiés

1. **Code dupliqué** : Même fonction `create_binary_generators()` dans 3 notebooks différents
2. **Notebooks trop lourds** : 200+ lignes de code de setup avant analyse
3. **Pas de réutilisation** : Impossible d'importer facilement une fonction
4. **Scripts isolés** : Chaque script réinvente la roue
5. **Structure non standard** : Pas conforme aux best practices ML

---

# ✅ ARCHITECTURE CIBLE

## Structure Professionnelle
```
zoidberg/
│
├── 📂 data/
│   ├── raw/                    # Données brutes (pas sur git)
│   │   └── chest_Xray/
│   └── processed/              # Données préprocessées (optionnel)
│
├── 📂 src/                     # ⭐ CODE PRINCIPAL (modules Python)
│   ├── __init__.py
│   │
│   ├── 📂 data/               # Chargement et préprocessing
│   │   ├── __init__.py
│   │   ├── loaders.py         # ImageDataGenerators, split
│   │   └── preprocessing.py   # Normalisation, augmentation
│   │
│   ├── 📂 models/             # Architectures et entraînement
│   │   ├── __init__.py
│   │   ├── cnn.py             # CNN from scratch
│   │   ├── efficientnet.py    # Transfer Learning
│   │   ├── pipeline.py        # Pipeline hiérarchique
│   │   └── training.py        # Fonctions d'entraînement
│   │
│   ├── 📂 evaluation/         # Métriques et évaluation
│   │   ├── __init__.py
│   │   ├── metrics.py         # AUC, Recall, F1, etc.
│   │   └── reports.py         # Génération rapports
│   │
│   ├── 📂 visualization/      # ✅ Déjà existant
│   │   ├── __init__.py
│   │   ├── data_exploration.py
│   │   ├── grad_cam.py
│   │   └── plots.py           # NOUVEAU: ROC, confusion matrix
│   │
│   └── 📂 utils/              # Utilitaires
│       ├── __init__.py
│       ├── config.py          # Configurations centralisées
│       └── helpers.py         # Fonctions helper
│
├── 📂 notebooks/              # ⭐ DÉMONSTRATION (légers, clairs)
│   ├── 01_data_exploration.ipynb      # Dataset overview
│   ├── 02_model_training.ipynb        # Entraînement modèles
│   ├── 03_evaluation.ipynb            # Évaluation + métriques
│   ├── 04_pipeline_demo.ipynb         # Pipeline hiérarchique
│   └── 05_MASTER.ipynb                # Notebook principal complet
│
├── 📂 scripts/                # Scripts autonomes (CLI)
│   ├── train_model.py         # Entraîner un modèle
│   ├── evaluate_model.py      # Évaluer un modèle
│   ├── generate_report.py     # Générer rapport
│   └── predict.py             # Prédiction sur nouvelle image
│
├── 📂 models/
│   ├── trained/               # ✅ Modèles sauvegardés
│   └── evaluation/            # ✅ Métriques JSON
│
├── 📂 reports/                # ✅ OK
│   ├── figures/
│   ├── COMPARATIVE_ANALYSIS.md
│   └── ...
│
├── 📂 tests/                  # NOUVEAU: Tests unitaires
│   ├── __init__.py
│   ├── test_data_loaders.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── 📄 config.yaml             # NOUVEAU: Configuration centralisée
├── 📄 requirements.txt        # ✅ OK
├── 📄 setup.py                # NOUVEAU: Installation package
├── 📄 README.md               # ✅ OK
└── 📄 .gitignore              # ✅ OK
```

---

# 📋 PLAN D'ACTION DÉTAILLÉ

## ÉTAPE 1 : Créer la Structure (30 min)

### Actions
```bash
# Créer les dossiers manquants
mkdir -p src/data
mkdir -p src/models
mkdir -p src/evaluation
mkdir -p src/utils
mkdir -p tests
mkdir -p data/processed

# Créer les __init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

### Fichiers à créer
- ✅ Structure de dossiers
- ✅ Fichiers `__init__.py` vides
- ✅ Fichier `config.yaml`
- ✅ Fichier `setup.py`

---

## ÉTAPE 2 : Migrer Code Data (1 heure)

### 2.1 Créer `src/data/loaders.py`

**Contenu** :
```python
"""
Module de chargement et génération de données.
Centralise toutes les fonctions de data loading.
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
                             img_size=DataConfig.IMG_SIZE,
                             batch_size=DataConfig.BATCH_SIZE,
                             val_split=DataConfig.VALIDATION_SPLIT):
    """
    Crée les générateurs pour classification binaire.
    
    Args:
        data_dir (str): Chemin vers les données
        img_size (tuple): Taille des images
        batch_size (int): Taille des batchs
        val_split (float): Proportion validation
        
    Returns:
        tuple: (train_gen, val_gen, test_gen)
    """
    if data_dir is None:
        data_dir = DataConfig.DATA_DIR
    
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
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen


def create_multiclass_generators(data_dir=None,
                                 img_size=DataConfig.IMG_SIZE,
                                 batch_size=DataConfig.BATCH_SIZE):
    """
    Crée les générateurs pour classification 3-classes.
    
    Args:
        data_dir (str): Chemin vers les données
        img_size (tuple): Taille des images
        batch_size (int): Taille des batchs
        
    Returns:
        tuple: (train_gen, val_gen, test_gen)
    """
    # Implémentation similaire
    # ...
    pass
```

**Bénéfices** :
- ✅ 1 seule source de vérité pour data loading
- ✅ Configuration centralisée
- ✅ Réutilisable dans tous les notebooks
- ✅ Documenté avec docstrings

---

### 2.2 Créer `src/data/preprocessing.py`

**Contenu** :
```python
"""
Module de preprocessing des données.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(y_train):
    """
    Calcule les poids de classes pour gérer déséquilibre.
    
    Args:
        y_train (array): Labels d'entraînement
        
    Returns:
        dict: {class_id: weight}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train
    )
    
    return dict(zip(classes, weights))


def normalize_image(image):
    """
    Normalise une image [0, 255] → [0, 1].
    
    Args:
        image (array): Image brute
        
    Returns:
        array: Image normalisée
    """
    return image / 255.0
```

---

## ÉTAPE 3 : Migrer Code Models (1 heure)

### 3.1 Créer `src/models/cnn.py`

**Contenu** :
```python
"""
Architecture CNN from scratch.
"""

from tensorflow import keras
from tensorflow.keras import layers, models


def build_cnn(input_shape=(128, 128, 3), num_classes=3):
    """
    Construit un CNN custom from scratch.
    
    Args:
        input_shape (tuple): Taille input (H, W, C)
        num_classes (int): Nombre de classes
        
    Returns:
        Model: Modèle Keras compilé
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

---

### 3.2 Créer `src/models/efficientnet.py`

**Contenu** :
```python
"""
Modèles EfficientNet avec Transfer Learning.
"""

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnet_binary(input_shape=(224, 224, 3),
                               learning_rate=1e-5,
                               fine_tune_layers=20):
    """
    Construit EfficientNet pour classification binaire.
    
    Args:
        input_shape (tuple): Taille input
        learning_rate (float): Learning rate
        fine_tune_layers (int): Nombre de couches à fine-tuner
        
    Returns:
        Model: Modèle compilé
    """
    # Base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base layers
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    
    # Build complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_efficientnet_multiclass(input_shape=(224, 224, 3),
                                  num_classes=3,
                                  learning_rate=1e-5):
    """
    Construit EfficientNet pour classification multi-classes.
    
    Args:
        input_shape (tuple): Taille input
        num_classes (int): Nombre de classes
        learning_rate (float): Learning rate
        
    Returns:
        Model: Modèle compilé
    """
    # Implémentation similaire
    # ...
    pass
```

---

### 3.3 Créer `src/models/pipeline.py`

**Contenu** :
```python
"""
Pipeline hiérarchique 2-stages.
"""

import numpy as np
from tensorflow import keras


class HierarchicalPipeline:
    """
    Pipeline de classification hiérarchique.
    
    Stage 1: Normal vs Pneumonie
    Stage 2: Bactérie vs Virus (si Pneumonie)
    """
    
    def __init__(self, 
                 stage1_model_path,
                 stage2_model_path,
                 threshold_stage1=0.5,
                 threshold_stage2=0.5):
        """
        Initialise le pipeline.
        
        Args:
            stage1_model_path (str): Chemin modèle stage 1
            stage2_model_path (str): Chemin modèle stage 2
            threshold_stage1 (float): Seuil décision stage 1
            threshold_stage2 (float): Seuil décision stage 2
        """
        self.stage1_model = keras.models.load_model(stage1_model_path)
        self.stage2_model = keras.models.load_model(stage2_model_path)
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2
    
    def predict(self, image):
        """
        Prédiction hiérarchique sur une image.
        
        Args:
            image (array): Image préprocessée
            
        Returns:
            tuple: (classe, confiance)
                classe: 'Normal', 'Bactérie', ou 'Virus'
                confiance: float entre 0 et 1
        """
        # Stage 1: Normal vs Pneumonie
        stage1_prob = self.stage1_model.predict(image, verbose=0)[0][0]
        
        if stage1_prob < self.threshold_stage1:
            # Normal
            return 'Normal', (1 - stage1_prob)
        
        # Stage 2: Bactérie vs Virus
        stage2_prob = self.stage2_model.predict(image, verbose=0)[0][0]
        
        if stage2_prob < self.threshold_stage2:
            return 'Bactérie', (1 - stage2_prob)
        else:
            return 'Virus', stage2_prob
    
    def predict_batch(self, images):
        """
        Prédiction sur un batch d'images.
        
        Args:
            images (array): Batch d'images
            
        Returns:
            list: [(classe, confiance), ...]
        """
        results = []
        for img in images:
            img_expanded = np.expand_dims(img, axis=0)
            result = self.predict(img_expanded)
            results.append(result)
        
        return results
```

---

## ÉTAPE 4 : Migrer Code Evaluation (45 min)

### 4.1 Créer `src/evaluation/metrics.py`

**Contenu** :
```python
"""
Calcul de métriques d'évaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_all_metrics(y_true, y_pred, y_proba=None):
    """
    Calcule toutes les métriques d'évaluation.
    
    Args:
        y_true (array): Vraies classes
        y_pred (array): Prédictions
        y_proba (array): Probabilités (pour AUC)
        
    Returns:
        dict: Toutes les métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
    }
    
    # AUC si probabilités disponibles
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc'] = None
    
    return metrics


def get_classification_report_dict(y_true, y_pred, class_names=None):
    """
    Génère un rapport de classification sous forme de dict.
    
    Args:
        y_true (array): Vraies classes
        y_pred (array): Prédictions
        class_names (list): Noms des classes
        
    Returns:
        dict: Rapport détaillé
    """
    return classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
```

---

## ÉTAPE 5 : Créer Notebooks Légers (1 heure)

### 5.1 Notebook `01_data_exploration.ipynb`

**Structure** :
```python
# Cellule 1: Imports
from src.data.loaders import DataConfig, create_binary_generators
from src.visualization.data_exploration import (
    show_class_distribution,
    visualize_batch
)

# Cellule 2: Chargement données
train_gen, val_gen, test_gen = create_binary_generators()

# Cellule 3: Distribution classes
show_class_distribution(train_gen)

# Cellule 4: Visualisation échantillons
visualize_batch(train_gen, num_images=9)

# → Total: 4 cellules au lieu de 30 !
```

**Bénéfices** :
- ✅ Code minimal dans notebook
- ✅ Imports clairs
- ✅ Focus sur résultats/analyses
- ✅ Réutilise code de `src/`

---

### 5.2 Notebook `02_model_training.ipynb`

**Structure** :
```python
# Cellule 1: Imports
from src.data.loaders import create_binary_generators
from src.models.efficientnet import build_efficientnet_binary
from src.data.preprocessing import compute_class_weights

# Cellule 2: Données
train_gen, val_gen, test_gen = create_binary_generators()

# Cellule 3: Modèle
model = build_efficientnet_binary(
    input_shape=(224, 224, 3),
    learning_rate=1e-5,
    fine_tune_layers=20
)

# Cellule 4: Class weights
class_weights = compute_class_weights(train_gen.classes)

# Cellule 5: Entraînement
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    class_weight=class_weights
)

# Cellule 6: Sauvegarde
model.save('models/trained/efficientnet_binary.keras')

# → Total: 6 cellules claires !
```

---

### 5.3 Notebook `05_MASTER.ipynb` (Notebook Principal)

**Structure** :
```markdown
# Projet Zoidberg - Notebook Principal

## 1. Introduction
- Abstract
- Objectifs
- Dataset overview

## 2. Modèle 1: CNN From Scratch
- Import: `from src.models.cnn import build_cnn`
- Entraînement minimal
- Résultats

## 3. Modèle 2: EfficientNet Binary
- Import: `from src.models.efficientnet import build_efficientnet_binary`
- Résultats

## 4. Modèle 3: EfficientNet Multi-classes
- Résultats

## 5. Modèle 4: Pipeline Hiérarchique
- Import: `from src.models.pipeline import HierarchicalPipeline`
- Démo

## 6. Comparaison
- Import: `from src.evaluation.metrics import compute_all_metrics`
- Tableau comparatif

## 7. Grad-CAM
- Import: `from src.visualization.grad_cam import generate_gradcam`

## 8. Conclusions
```

**Durée** : 30 minutes à créer (réutilise tout le code existant)

---

## ÉTAPE 6 : Configuration Centralisée (30 min)

### 6.1 Créer `config.yaml`

**Contenu** :
```yaml
# Configuration Projet Zoidberg

# Données
data:
  raw_dir: "data/raw/chest_Xray"
  processed_dir: "data/processed"
  img_size: [224, 224]
  batch_size: 32
  validation_split: 0.2

# Classes
classes:
  names: ["Normal", "Bactérie", "Virus"]
  binary_names: ["Normal", "Pneumonie"]

# Entraînement
training:
  epochs: 50
  learning_rate: 1e-5
  early_stopping_patience: 5
  
  # Fine-tuning
  fine_tune_layers: 20
  
  # Data augmentation
  augmentation:
    rotation_range: 20
    zoom_range: 0.2
    horizontal_flip: true
    brightness_range: [0.8, 1.2]

# Modèles
models:
  cnn_scratch:
    input_shape: [128, 128, 3]
    num_classes: 3
  
  efficientnet:
    input_shape: [224, 224, 3]
    weights: "imagenet"

# Evaluation
evaluation:
  test_dir: "data/raw/chest_Xray/test"
  metrics: ["accuracy", "precision", "recall", "f1", "auc"]

# Chemins de sauvegarde
paths:
  models: "models/trained"
  evaluation: "models/evaluation"
  reports: "reports"
  figures: "reports/figures"
```

---

### 6.2 Créer `src/utils/config.py`

**Contenu** :
```python
"""
Utilitaire pour charger la configuration.
"""

import yaml
from pathlib import Path


class Config:
    """Classe de configuration singleton."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """Charge le fichier config.yaml."""
        config_path = Path("config.yaml")
        with open(config_path, 'r') as f:
            cls._config = yaml.safe_load(f)
    
    @classmethod
    def get(cls, key, default=None):
        """
        Récupère une valeur de configuration.
        
        Args:
            key (str): Clé (ex: 'data.img_size')
            default: Valeur par défaut
            
        Returns:
            Value ou default
        """
        keys = key.split('.')
        value = cls._config
        
        for k in keys:
            value = value.get(k, default)
            if value is default:
                break
        
        return value


# Usage dans les modules:
# from src.utils.config import Config
# img_size = Config.get('data.img_size')
```

---

## ÉTAPE 7 : Setup.py pour Installation (15 min)

### Créer `setup.py`

**Contenu** :
```python
"""
Setup pour installer le package Zoidberg.
"""

from setuptools import setup, find_packages

setup(
    name='zoidberg',
    version='1.0.0',
    description='Détection de Pneumonie par Deep Learning',
    author='Nicolas BRODBECK',
    author_email='nbrodbeck@email.com',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.21.0',
        'numpy>=2.1.0',
        'pandas>=2.2.3',
        'scikit-learn>=1.5.2',
        'matplotlib>=3.9.2',
        'seaborn>=0.13.2',
        'pillow>=11.0.0',
        'pyyaml>=6.0'
    ],
    python_requires='>=3.9',
)
```

**Installation** :
```bash
# En mode développement (éditable)
pip install -e .

# Maintenant tu peux importer partout:
from src.data.loaders import create_binary_generators
from src.models.efficientnet import build_efficientnet_binary
```

---

## ÉTAPE 8 : Tests Unitaires (30 min - Optionnel)

### Créer `tests/test_data_loaders.py`

**Contenu** :
```python
"""
Tests pour les data loaders.
"""

import unittest
from src.data.loaders import create_binary_generators, DataConfig


class TestDataLoaders(unittest.TestCase):
    
    def test_create_binary_generators(self):
        """Test création générateurs binaires."""
        train_gen, val_gen, test_gen = create_binary_generators()
        
        # Vérifier que les générateurs sont créés
        self.assertIsNotNone(train_gen)
        self.assertIsNotNone(val_gen)
        self.assertIsNotNone(test_gen)
        
        # Vérifier la taille des batchs
        self.assertEqual(train_gen.batch_size, DataConfig.BATCH_SIZE)
        
        # Vérifier les classes
        self.assertEqual(train_gen.num_classes, 2)  # Binaire
    
    def test_config_values(self):
        """Test valeurs de configuration."""
        self.assertEqual(DataConfig.IMG_SIZE, (224, 224))
        self.assertEqual(DataConfig.BATCH_SIZE, 32)


if __name__ == '__main__':
    unittest.main()
```

**Lancer tests** :
```bash
python -m unittest discover tests/
```

---

# ✅ RÉSULTAT FINAL

## Avant vs Après

### ❌ AVANT (Code dupliqué)
```python
# Dans experiments_v3.ipynb (200 lignes de setup)
def create_binary_generators(...):
    # 50 lignes de code
    
# Dans experiments_v3_FIXED.ipynb (encore 50 lignes)
def create_binary_generators(...):
    # Code identique dupliqué
    
# Dans 07_comprehensive_analysis.ipynb
# Encore 30 lignes de setup
```

### ✅ APRÈS (Code modulaire)
```python
# Dans TOUS les notebooks (1 ligne)
from src.data.loaders import create_binary_generators

# C'est tout ! 🎉
```

---

## Avantages de la Nouvelle Structure

### 1. ✅ Réutilisabilité
```python
# Import facile partout
from src.models.pipeline import HierarchicalPipeline

# Utilisation
pipeline = HierarchicalPipeline(
    'models/trained/stage1.keras',
    'models/trained/stage2.keras'
)
prediction = pipeline.predict(image)
```

### 2. ✅ Notebooks Légers
```
Avant : 200+ lignes de code
Après : 20-30 lignes de code

Focus : Résultats et analyses, pas implémentation
```

### 3. ✅ Tests Unitaires
```bash
# Vérifier que tout marche
python -m unittest discover tests/

# ✅ 10 tests passed
```

### 4. ✅ Documentation Automatique
```bash
# Générer documentation avec Sphinx
sphinx-apidoc -o docs/ src/
sphinx-build docs/ docs/_build

# → Documentation HTML professionnelle
```

### 5. ✅ Import depuis n'importe où
```bash
# Installer en mode développement
pip install -e .

# Maintenant dans un script Python PARTOUT:
from src.models.efficientnet import build_efficientnet_binary
```

---

# 🚀 MIGRATION PROGRESSIVE

## Option 1 : Migration Complète (4h)
**Recommandée si** : Tu as le temps et veux un projet ultra-pro

1. Créer toute la structure (30 min)
2. Migrer tout le code (2h30)
3. Créer notebooks légers (1h)
4. Tests (30 min optionnel)

**Résultat** : Projet 100% professionnel

---

## Option 2 : Migration Minimale (1h30) ⭐ **RECOMMANDÉE**
**Recommandée si** : Tu veux améliorer sans tout casser

1. Créer `src/data/loaders.py` (30 min)
2. Créer `src/models/efficientnet.py` (30 min)
3. Créer `src/models/pipeline.py` (30 min)
4. Mettre à jour 1-2 notebooks pour montrer l'utilisation

**Résultat** : Amélioration visible, code réutilisable

---

## Option 3 : Juste Config (30 min)
**Recommandée si** : Tu veux juste centraliser

1. Créer `config.yaml` (15 min)
2. Créer `src/utils/config.py` (15 min)
3. Documenter dans README

**Résultat** : Configuration centralisée

---

# 📋 CHECKLIST MIGRATION

## Avant de commencer
- [ ] Commit actuel sur git (backup)
- [ ] Créer branche `restructure-project`
- [ ] Lire ce plan complètement

## Structure
- [ ] Créer dossiers `src/data`, `src/models`, `src/evaluation`, `src/utils`
- [ ] Créer tous les `__init__.py`
- [ ] Créer `config.yaml`
- [ ] Créer `setup.py`

## Migration Code
- [ ] Créer `src/data/loaders.py`
- [ ] Créer `src/data/preprocessing.py`
- [ ] Créer `src/models/cnn.py`
- [ ] Créer `src/models/efficientnet.py`
- [ ] Créer `src/models/pipeline.py`
- [ ] Créer `src/evaluation/metrics.py`
- [ ] Créer `src/utils/config.py`

## Notebooks
- [ ] Créer ou mettre à jour `01_data_exploration.ipynb`
- [ ] Créer ou mettre à jour `02_model_training.ipynb`
- [ ] Créer `05_MASTER.ipynb`
- [ ] Tester que tous les imports fonctionnent

## Tests
- [ ] Créer `tests/test_data_loaders.py`
- [ ] Lancer tests unitaires
- [ ] Tout passe ✅

## Documentation
- [ ] Mettre à jour README.md
- [ ] Ajouter instructions installation
- [ ] Documenter nouvelle structure

## Git
- [ ] Commit changements
- [ ] Merge dans main
- [ ] Push sur GitHub

---

# 🎯 RECOMMANDATION FINALE

**Pour toi, je recommande OPTION 2 (1h30)** :

1. ✅ Gain immédiat visible
2. ✅ Code réutilisable pour oral
3. ✅ Pas trop de changements (risque faible)
4. ✅ Notebooks plus professionnels
5. ✅ Montre architecture logicielle au jury

**Tu peux faire Option 1 après l'oral** si tu veux aller plus loin.

---

**Tu veux qu'on commence la restructuration ensemble ?** 🏗️

Je peux te guider étape par étape ! 💪
