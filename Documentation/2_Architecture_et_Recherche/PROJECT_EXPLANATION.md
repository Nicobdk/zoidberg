# 📚 Documentation Technique - Projet Zoidberg

Ce document détaille la structure et le fonctionnement du projet **Zoidberg**, une application de classification d'images médicales (radiographies thoraciques) visant à détecter la pneumonie et son origine (Virale ou Bactérienne).

## 📂 Vue d'ensemble des fichiers

Le projet est structuré autour de plusieurs modules Python, chacun ayant une responsabilité spécifique : chargement des données, exploration/visualisation, définition des modèles, interprétabilité et exécution principale.

---

### 1. `main.py`

C'est le **point d'entrée** de l'application. Il orchestre l'ensemble du flux de travail classique.

- **Rôle** :
  - Initialise les générateurs de données.
  - Calcule les poids des classes (`class_weights`) pour gérer le déséquilibre éventuel entre les classes.
  - Construit et compile le modèle CNN ou Transfert.
  - Lance l'entraînement du modèle sur les données d'entraînement.
  - Évalue le modèle sur les données de test et affiche des visualisations.

---

### 2. `src/data/data_loader_v3.py`

Ce fichier gère le **chargement et le prétraitement des données** de manière optimisée. L'ancien script `data_loader.py` a été supprimé pour éviter la confusion.

- **Fonction principale** : `create_data_generator(data_dir)`
- **Détails** :
  - Utilise `ImageDataGenerator` de Keras pour charger les images depuis les dossiers.
  - Applique une **normalisation** (rescale 1./255) pour mettre les pixels à l'échelle [0, 1].
  - Implémente de l'**augmentation de données** (data augmentation) sur l'ensemble d'entraînement pour éviter le surapprentissage :
    - Rotation, Zoom, et Retournement horizontal.
  - **Note** : Le pipeline inclut des stratégies de "Center Cropping" et de prétraitement spécifiques aux modèles de Transfer Learning (comme `preprocess_input` de EfficientNet).

---

### 3. `src/models/` : L'Évolution Architecturale

Le dossier des modèles est le cœur de Zoidberg. Il a grandement évolué depuis la simple baseline pour intégrer des pipelines complexes.

#### 3.1. Modèles de base
- **`baseline_model.py`** : Modèle de référence (PCA + Régression Logistique). Sert de point de comparaison "Machine Learning classique".
- **`cnn_model.py`** : Un CNN binaire créé "from scratch" (Conv2D, MaxPooling, BatchNorm) très performant en détection pure (Recall 98%).

#### 3.2. L'Approche Transfer Learning
- **`binary_model.py`** : Modèle de détection de pneumonie basé sur **EfficientNetB0** pré-entraîné (ImageNet). Les 20 dernières couches sont fine-tunées.
- **`multiclass_model.py` & `transfert_model.py`** : Modèles utilisant également EfficientNetB0 mais avec une activation **Softmax** finale pour diviser les probabilités entre 3 classes (Normal, Bactérie, Virus).

#### 3.3. L'Approche Hiérarchique (Pipeline Avancé)
- **`hierarchical_pipeline.py`** : Classe `HierarchicalClassifier` qui orchestre deux modèles à la suite :
  1. *Étape 1* : Le modèle binaire vérifie si le patient est Normal ou Atteint de Pneumonie.
  2. *Étape 2* : S'il est atteint, le modèle multi-classes (ou sous-type) prend le relais pour déterminer s'il s'agit d'une pneumonie Virale ou Bactérienne.
- **`evaluate_hierarchical.py`** : Script dédié à l'évaluation de ce pipeline sur le jeu de test complet avec génération de matrice de confusion médicale.

---

### 4. `src/visualization/` : Interprétabilité et Exploration

La visualisation est cruciale dans le domaine médical pour justifier les décisions de l'IA (Explainable AI).

#### 4.1. `data_exploration.py`
Ce fichier contient des outils classiques pour comprendre les données :
- Affichage de la distribution des classes (graphiques à barres).
- Affichage d'un lot d'images avec leurs étiquettes.

#### 4.2. `grad_cam.py` (Nouveau module critique)
Afin de contrer le biais de "Shortcut Learning" (quand l'IA regarde les lettres sur la radio au lieu du poumon), ce script a été développé :
- **Fonctions clés** : `make_gradcam_heatmap`, `display_gradcam`, `find_last_conv_layer`.
- **Rôle** : Calcule le gradient de la prédiction par rapport à la dernière couche de convolution pour générer une **Heatmap** (carte de chaleur). Cette carte est superposée à la radio pour voir **exactement quelles zones anatomiques l'IA a "regardé"** pour prendre sa décision.

---

### 5. `src/models/evaluation.py`

Ce module regroupe les fonctions analytiques post-entraînement.

- **Rôle** : Produit les métriques médicales clés.
- **Fonctions** :
  - Génération des rapports de classification complets (Accuracy, Precision, Recall, F1-Score).
  - Tracé et sauvegarde des **Matrices de Confusion**.
  - Trace des courbes d'apprentissage (Loss/Accuracy sur les epochs).

---

## 🛠 Résumé des dépendances principales

- **TensorFlow / Keras** : Moteur Deep Learning principal (Entraînement, Transfer Learning, Grad-CAM).
- **Scikit-learn** : Modèle baseline (PCA, LogisticRegression) et métriques (classification_report, confusion_matrix).
- **Matplotlib & Seaborn** : Visualisation des données et Heatmaps avancées.
- **NumPy** : Opérations mathématiques matricielles sous-jacentes.
