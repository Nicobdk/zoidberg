# 🫁 ZOIDBERG — Pneumonia Detection with Deep Learning

## 📌 Project Overview

**ZOIDBERG** est un projet de classification d'images médicales visant à assister la détection de la pneumonie à partir de radiographies thoraciques (X-ray) en utilisant des techniques de Machine Learning et de Deep Learning.

Le projet suit une méthodologie scientifique structurée :

1.  **Exploration** et compréhension des données.
2.  **Baseline** : Comparaison avec des modèles classiques.
3.  **CNN from scratch** : Développement d'un réseau de neurones convolutif.
4.  **Évaluation médicale** : Utilisation de métriques de performance avancées.

---

## 🎯 Objectives

Construire un modèle de classification robuste capable de distinguer :

- **Poumons sains** (Normal)
- **Cas de pneumonie** (Pneumonia)

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

## 📈 Evaluation Metrics

Le modèle démontre un fort pouvoir de discrimination grâce aux outils suivants :

- **Matrice de Confusion**
- **Précision / Rappel (Recall)**
- **F1-score**
- **Courbe ROC & ROC-AUC**

---

## 📂 Project Structure

```text
.
├── data_loader.py       # Chargement et prétraitement des données
├── baseline_model.py    # Modèle ML de référence (PCA + LogReg)
├── cnn_model.py         # Script du modèle CNN principal
├── evaluation.py        # Fonctions de calcul des métriques
├── main.py              # Point d'entrée pour l'entraînement
└── requirements.txt     # Dépendances Python
```

## ⚙️ Installation & Dataset

```text
pip install -r requirements.txt
```

**Dataset :**

_Le dataset n'est pas inclus dans le dépôt._

Structure attendue :

```Plaintext

chest_Xray/
    train/
        normal/
        pneumonia/
    test/
        ...
```

## 🚀 Next Steps

- [x] Classification Multi-classe (Normal / Viral / Bactérien).

- [ ] Optimisation du seuil de décision.

- [ ] Transfer Learning (ResNet, EfficientNet).

- [ ] Interprétabilité (Grad-CAM pour voir ce que le modèle "regarde")
