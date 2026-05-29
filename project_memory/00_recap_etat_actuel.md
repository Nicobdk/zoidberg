# État Actuel du Projet Zoidberg (Récapitulatif)

Date: 20 avril 2026

## 🎯 Objectif Général
Le projet Zoidberg est une application de Deep Learning appliquée à l'imagerie médicale. Son but est de classifier des radiographies thoraciques (X-Rays) pour identifier la présence d'une pneumonie et, le cas échéant, de différencier si elle est d'origine **Virale** ou **Bactérienne**.

---

## 🧠 Les Modèles de Classification

Le projet implémente pour l'instant plusieurs niveaux de modèles, de la "baseline" mathématique basique jusqu'aux réseaux de neurones pré-entraînés complexes :

### 1. Modèle Baseline / Référence (`src/models/baseline_model.py`)
- **Algorithme** : Principal Component Analysis (PCA) suivie d'une Régression Logistique.
- **Principe** : Les images sont aplaties en 1D. La PCA permet de réduire le nombre de variables (réduction de dimension) pour ne garder que les 50 composantes principales.
- **Objectif** : Avoir un point de comparaison "Machine Learning classique" très basique, pour vérifier que le Deep Learning apporte bien une vraie valeur ajoutée.

### 2. CNN "From Scratch" - Classification Binaire (`src/models/cnn_model.py`)
- **Algorithme** : Réseau de Neurones Convolutif (CNN) créé de zéro, implémentant des blocs "Conv2D + BatchNorm + ReLU + MaxPooling".
- **Principe** : Effectue une classification binaire (**Normal vs Pneumonie** globale).
- **Architecture** : Termine par une couche *Dense* d'activation fonction Sigmoïd pour sortir une probabilité. 
- **Performances** : Atteint un très bon score en milieu médical avec un Recall d'environ **98%** (capacité à ne rater presque aucun malade) et une AUC de 0.93.

### 3. Transfer Learning - Classification Binaire (`src/models/binary_model.py`)
- **Algorithme** : Modèle basé sur **EfficientNetB0**.
- **Principe** : Ce modèle utilise la puissance de "Transfer Learning" à partir de poids initialisés sur ImageNet. 
- **Architecture** : Les couches du modèle de base sont gelées, à l'exception des 20 dernières qui sont "fine-tunées" sur les radios du projet. Se conclut par un GlobalAveragePooling, du Dropout et un neurone Dense en Sigmoïd.

### 4. Transfer Learning - Classification Multi-classes (`src/models/transfert_model.py`)
- **Algorithme** : Modèle basé sur **EfficientNetB0**.
- **Principe** : L'évolution du projet, effectuant une classification à trois choix : **Normal, Bactérie, ou Virus**.
- **Architecture** : Même procédure de "Fine-Tuning" des 20 dernières couches que ci-dessus, mais se termine par une activation locale **Softmax** pour diviser la probabilité entre les 3 classes.
- **Performances** : Rencontre actuellement des difficultés avec une précision générale et un Macro F1-Score autour de **55%-56%**. La distinction virale/bactérienne nécessite d'être optimisée ou d'obtenir plus de données (déséquilibre de classe).

---

## ⚙️ Algorithmes et Traitement de Données

Afin de permettre à ces modèles de comprendre au mieux la radiologie pulmonaire, plusieurs techniques algorithmiques sont déployées :

1. **Augmentation de données temporelle (Data Augmentation)** :
   Dans les "data_loaders", les algorithmes de Keras tournent l'image, appliquent des zooms et des retournements horizontaux aléatoires pour créer de nouvelles données d'entraînement virtuelles et limiter le surapprentissage (*Overfitting*).

2. **Pondération des classes (Class Weights)** :
   Pour compenser le "déséquilibre de classe" (imbalance) du dataset (par exemple, s'il y a plus d'images de bactéries que de poumons sains ou de cas viraux), un algorithme pondère la fonction de perte au moment de l'entraînement. Ainsi, une erreur du modèle sur une classe "rare" est punie plus sévèrement.

3. **Fonctions de perte (Loss Metrics)** :
   - Binary Crossentropy : Algorithme utilisé pour calculer l'erreur lors des distinctions binaire (Normal vs Malade).
   - Categorical Crossentropy : Algorithme appliqué pour les calculs d'erreur Multi-classes (Normal vs Viral vs Bactérien).

4. **Optimisation Mathématique** :
   Le projet s'appuie sur l'optimiseur **Adam** avec des "callbacks" critiques :
   - `ReduceLROnPlateau` : Si la stagnation de l'apprentissage est détectée, cet algo baisse automatiquement le "Learning Rate" pour être plus concis et précis et débloquer la phase d'apprentissage.
   - `EarlyStopping` : Stoppe automatiquement le modèle lorsque les performances sur les données de validation commencent à dégrader.

## 🔍 Interprétabilité et Validation Médicale

Pour garantir que le modèle prend ses décisions sur des critères médicaux réels et non sur des artefacts :

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)** :
   Implémenté dans `src/visualization/grad_cam.py`, cet algorithme permet de générer des "heatmaps" d'attention. Il met en évidence les zones de l'image qui ont le plus influencé la prédiction du modèle.

2. **Lutte contre le "Shortcut Learning"** :
   Grâce à Grad-CAM, nous avons identifié que le modèle avait tendance à "tricher" en regardant les lettres (L/R) ou les bords des images. Pour corriger cela, une stratégie de **Center Cropping** a été adoptée, recentrant l'analyse sur le parenchyme pulmonaire.

---

> Ce fichier fait partie du **Project Memory Zoidberg**. Il sera conservé et mis à jour comme un point de sauvegarde chronologique.
