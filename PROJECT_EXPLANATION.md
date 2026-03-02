# 📚 Documentation Technique - Projet Zoidberg

Ce document détaille la structure et le fonctionnement du projet **Zoidberg**, une application de classification d'images médicales (radiographies thoraciques) visant à détecter la pneumonie.

## 📂 Vue d'ensemble des fichiers

Le projet est structuré autour de plusieurs modules Python, chacun ayant une responsabilité spécifique : chargement des données, exploration/visualisation, définition des modèles et exécution principale.

---

### 1. `main.py`

C'est le **point d'entrée** de l'application. Il orchestre l'ensemble du flux de travail.

- **Rôle** :
  - Initialise les générateurs de données via `data_loader.py`.
  - Calcule les poids des classes (`class_weights`) pour gérer le déséquilibre éventuel entre les classes (Normal vs Pneumonia).
  - Construit et compile le modèle CNN via `cnn_model.py`.
  - Lance l'entraînement du modèle sur les données d'entraînement.
  - Évalue le modèle sur les données de test.
  - Affiche des visualisations (batch d'images, distribution des classes).

---

### 2. `data_loader.py`

Ce fichier gère le **chargement et le prétraitement des données**.

- **Fonction principale** : `create_data_generator(data_dir)`
- **Détails** :
  - Utilise `ImageDataGenerator` de Keras pour charger les images depuis les dossiers.
  - Applique une **normalisation** (rescale 1./255) pour mettre les pixels à l'échelle [0, 1].
  - Applique de l'**augmentation de données** (data augmentation) sur l'ensemble d'entraînement pour éviter le surapprentissage :
    - Rotation (`rotation_range`)
    - Zoom (`zoom_range`)
    - Retournement horizontal (`horizontal_flip`)
  - Retourne trois générateurs : `train_gen` (entraînement), `val_gen` (validation) et `test_gen` (test).

---

### 3. `cnn_model.py`

Ce fichier définit l'architecture du **Réseau de Neurones Convolutif (CNN)**.

- **Fonction principale** : `build_cnn(input_shape)`
- **Architecture** :
  - Modèle séquentiel Keras.
  - Composé de plusieurs blocs de convolution (`Conv2D`), normalisation (`BatchNormalization`), activation (`ReLU`) et pooling (`MaxPooling2D`).
  - Inclut une couche de `Dropout` (0.5) pour la régularisation.
  - Se termine par des couches denses (`Dense`) pour la classification.
  - La sortie utilise une activation **Sigmoid** pour une classification binaire (0 ou 1).
- **Compilation** : Utilise l'optimiseur `Adam` et la perte `binary_crossentropy`.

---

### 4. `baseline_model.py`

Ce fichier propose un **modèle de référence (baseline)** simple pour comparer les performances.

- **Fonction principale** : `train_baseline(...)`
- **Approche** :
  - **PCA (Principal Component Analysis)** : Réduit la dimensionnalité des images en ne gardant que 50 composantes principales.
  - **Régression Logistique** : Un classifieur linéaire simple entraîné sur les données réduites par PCA.
- **Objectif** : Fournir un point de comparaison "simple" pour voir si le CNN (plus complexe) apporte réellement une amélioration.

---

### 5. `data_exploration.py`

Ce fichier contient des **outils de visualisation** pour mieux comprendre les données.

- **Fonctions clés** :
  - `show_class_distribution(generator)` : Affiche un graphique à barres montrant le nombre d'images "Normal" vs "Pneumonia".
  - `visualize_batch(generator, n)` : Affiche un lot d'images avec leurs étiquettes.
  - `visualize_specific_class(...)` : Permet de visualiser des images d'une classe spécifique.

---

### 6. `evaluation.py`

- **État actuel** : Fichier vide.
- **Usage prévu** : Destiné à contenir des fonctions avancées pour évaluer les modèles (matrices de confusion, courbes ROC/AUC, rapports de classification détaillés).

---

## 🛠 Résumé des dépendances principales

- **TensorFlow / Keras** : Pour la construction et l'entraînement du CNN.
- **Scikit-learn** : Pour le modèle baseline (PCA, LogisticRegression) et le calcul des poids des classes.
- **Matplotlib** : Pour la visualisation des données.
- **NumPy** : Pour les opérations mathématiques sur les tableaux.
