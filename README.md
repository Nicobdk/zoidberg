# 🫁 ZOIDBERG — Pneumonia Detection with Deep Learning

## 📌 Project Overview

**ZOIDBERG** est un projet de classification d'images médicales visant à assister la détection de la pneumonie à partir de radiographies thoraciques (X-ray) en utilisant des techniques de Machine Learning et de Deep Learning.

> [!TIP]
> **Nouveau dans la documentation ?** Consultez le **[Guide de la Documentation](./project_memory/01_guide_documentation.md)** pour un récapitulatif global et une carte de tous les fichiers de doc.

Le projet suit une méthodologie scientifique structurée :

1.  **Exploration** et compréhension des données.
2.  **Baseline** : Comparaison avec des modèles classiques.
3.  **CNN from scratch** : Développement d'un réseau de neurones convolutif.
4.  **Évaluation médicale** : Utilisation de métriques de performance avancées.

---

## 🎯 Objectives

Construire un modèle de classification robuste capable de distinguer :

- **Poumons sains** (Normal)
- **Pneumonie Bactérienne** (Bacterial Pneumonia)
- **Pneumonie Virale** (Viral Pneumonia)

**Historique :** Le projet a d'abord commencé par une classification binaire (Normal vs Pneumonia globale) avant d'évoluer vers une classification multi-classes.

**Points d'attention particuliers :**

- Gestion du déséquilibre des classes (Class Imbalance).
- Méthodologie de validation rigoureuse.
- Focus sur les métriques médicales (Recall, ROC-AUC).

---

## 🧠 Models Implemented

### 1️⃣ Baseline Model (Machine Learning)

- **Méthode** : Images aplaties (Flattened).
- **Réduction de dimension** : PCA (Principal Component Analysis).
- **Classifieur** : Régression Logistique.
- **But** : Établir une base de comparaison pour le Deep Learning.

### 2️⃣ CNN (Main Model)

**Architecture :**

- **Input** : 128x128x3.
- **Blocs Convolutifs** : Conv + BatchNorm + ReLU + MaxPooling.
- **Régularisation** : Dropout.
- **Couche Dense** : 128 neurones.
- **Sortie** : Sigmoid (Classification binaire).

[Image of CNN architecture diagram]

**Optimisation de l'entraînement :**

- **Class Weighting** : Pour compenser le déséquilibre des données.
- **EarlyStopping** : Pour éviter le sur-apprentissage.
- **ReduceLROnPlateau** : Ajustement dynamique du Learning Rate.

### 3️⃣ Transfer Learning (Multiclass Model)

**Architecture :**

- **Base Model** : EfficientNetB0 (pré-entraîné sur ImageNet).
- **Modification** : Gel de la majorité des couches, `Fine-tuning` sur les 20 dernières couches.
- **Classification Head** : GlobalAveragePooling2D + BatchNormalization + Dropout (0.5) + Dense (128) + Dropout (0.3).
- **Sortie** : Softmax pour la classification multi-classes (3 classes : Normal, Bacteria, Virus).

**Optimisation :**

- **Optimiseur** : Adam avec un faible Learning Rate (1e-5).
- **Perte** : Categorical Crossentropy.
- **Preprocessing** : Prétraitement natif à EfficientNet (`preprocess_input`).

---

## 📊 Final Results (Binary Classification)

| Métrique                | Score      |
| :---------------------- | :--------- |
| **Train Accuracy**      | ~ 95–96%   |
| **Validation Accuracy** | ~ 90%      |
| **Test Accuracy**       | ~ 85%      |
| **Recall (Pneumonia)**  | **~ 98%**  |
| **AUC**                 | **~ 0.93** |

> 💡 **Key Insight** : Le modèle priorise fortement la détection des cas de pneumonie (Recall élevé), ce qui est critique pour éviter de rater un patient malade (faux négatifs).

---

## 📊 Final Results (Multiclass Classification)

Pour l'approche multi-classes (Normal vs Viral vs Bactérien) basée sur l'EfficientNet :

| Métrique              | Score |
| :-------------------- | :---- |
| **Accuracy Générale** | ~ 56% |
| **Macro F1-Score**    | ~ 55% |

> 💡 **Observation** : La distinction entre la pneumonie virale et bactérienne s'avère plus complexe que la classification binaire. Des optimisations sur le seuil de décision ou l'augmentation de données ciblée pourraient améliorer ces résultats.

---

## 📈 Evaluation Metrics

Le modèle démontre un fort pouvoir de discrimination grâce aux outils suivants :

- **Matrice de Confusion**
- **Précision / Rappel (Recall)**
- **F1-score**
- **Courbe ROC & ROC-AUC**

---

## 📂 Project Structure

L'architecture du projet a été refondue pour suivre les standards en Data Science / MLOps :

```text
.
├── data/
│   ├── raw/           # Données brutes originales (Dataset)
│   ├── interim/       # Données en cours de transformation
│   └── processed/     # Données finales prêtes pour la modélisation
├── notebooks/         # Notebooks Jupyter
│   ├── 01_exploration/
│   ├── 02_preprocessing/
│   └── 03_modeling/
├── src/               # Code source
│   ├── data/          # Scripts de dataloader (e.g., data_loader_v3.py)
│   ├── features/      # Feature engineering
│   ├── models/        # Entraînement et modélisation (cnn_model.py, transfert_model.py)
│   └── visualization/ # Génération de graphiques (data_exploration.py)
├── experiments/       # Fichiers de configuration & tracking
├── models/
│   ├── trained/       # Modèles sérialisés (.h5, .keras)
│   └── evaluation/    # Métriques (classification_reports.json)
├── reports/
│   └── figures/       # Visualisations & courbes d'entraînement
├── tests/             # Tests unitaires
├── .env.example       # Template des variables d'environnement
├── main.py            # Point d'entrée pour l'entraînement
├── requirements.txt   # Dépendances Python
└── README.md          # Doc du projet
```

## ⚙️ Installation & Dataset

```text
pip install -r requirements.txt
```

**Dataset :**

_Le dataset n'est pas inclus dans le dépôt, déposez-le dans `data/raw/`._

Structure attendue :

```Plaintext

data/
└── raw/
    └── chest_Xray/
        ├── train/
        │   ├── normal/
        │   └── pneumonia/
        └── test/...
```

## 🚀 Next Steps

- [x] Classification Multi-classe (Normal / Viral / Bactérien).

- [ ] Optimisation du seuil de décision.

- [x] Transfer Learning (ResNet, EfficientNet).

- [ ] Interprétabilité (Grad-CAM pour voir ce que le modèle "regarde")
