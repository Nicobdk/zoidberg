# 🎓 GUIDE PÉDAGOGIQUE COMPLET - PROJET ZOIDBERG
## Comprendre le Deep Learning pour la Détection de Pneumonie

**Auteur** : Nicolas BRODBECK  
**Pour** : Préparation à l'oral  
**Objectif** : Comprendre TOUT le projet de A à Z

---

# 📚 TABLE DES MATIÈRES

1. [Le Problème Médical](#1-le-problème-médical)
2. [Deep Learning : Les Bases](#2-deep-learning--les-bases)
3. [CNN : Comment ça Marche ?](#3-cnn--comment-ça-marche-)
4. [Transfer Learning Expliqué](#4-transfer-learning-expliqué)
5. [Ton Pipeline Hiérarchique](#5-ton-pipeline-hiérarchique)
6. [Les Métriques Médicales](#6-les-métriques-médicales)
7. [Grad-CAM et Explicabilité](#7-grad-cam-et-explicabilité)
8. [Résumé pour l'Oral](#8-résumé-pour-loral)

---

# 1. Le Problème Médical

## 🏥 Contexte Clinique

**Pneumonie** = Infection pulmonaire grave
- **200 millions** de cas/an dans le monde
- Peut être **mortelle** si non traitée rapidement

## 🎯 Le Défi Médical

Un radiologue regarde une **radio du thorax** (X-ray) et doit répondre à **3 questions** :

1. **Le poumon est-il sain ?** → Normal
2. **Si malade, c'est une bactérie ?** → Pneumonie Bactérienne
3. **Ou un virus ?** → Pneumonie Virale

**Pourquoi c'est important ?**
- Bactérie → Antibiotiques
- Virus → Antiviraux (ou repos)

**Mauvais traitement = danger pour le patient**

## 💡 L'Idée de Ton Projet

> "Et si une IA pouvait **aider** le radiologue à détecter plus vite et plus précisément ?"

**Ton rôle** : Créer cette IA avec du Deep Learning

---

# 2. Deep Learning : Les Bases

## 🧠 C'est Quoi un Réseau de Neurones ?

### Analogie Simple : Le Cerveau Humain

Imagine que tu apprends à reconnaître un chat :

**Bébé (0 neurones entraînés)**
- Tu vois une boule de poils → "???"

**Enfant (quelques neurones activés)**
- Parents : "Ça c'est un chat !"
- Ton cerveau **ajuste ses connexions**
- Tu vois 10, 20, 50 chats → Tu commences à reconnaître

**Adulte (neurones bien entraînés)**
- Tu vois une photo floue → "C'est un chat !"
- Même si c'est un chat bizarre, tu le reconnais

**➡️ Un réseau de neurones fait PAREIL**

### Structure d'un Réseau de Neurones

```
INPUT        HIDDEN LAYERS        OUTPUT
(Image)      (Traitement)         (Prédiction)

[Pixels] → [Neurone] → [Neurone] → [Résultat]
           [Neurone]    [Neurone]
           [Neurone]    [Neurone]
```

Chaque **neurone** :
1. Reçoit des infos (= inputs)
2. Les **pèse** avec des poids (= weights)
3. Fait un calcul
4. Donne un résultat

### Formule Mathématique d'un Neurone

```
y = f(W·x + b)
```

**Décortiquons** :
- `x` = Input (ce qui entre, ex: pixels)
- `W` = Weights (poids, les connexions)
- `b` = Bias (biais, un décalage)
- `W·x` = Multiplication matricielle (on pondère)
- `f()` = Fonction d'activation (voir après)
- `y` = Output (ce qui sort)

**Exemple Concret** :

Tu as 3 pixels : `x = [0.2, 0.5, 0.8]`

Les poids : `W = [0.3, 0.1, 0.4]`

Le biais : `b = 0.1`

**Calcul** :
```
W·x = (0.3 × 0.2) + (0.1 × 0.5) + (0.4 × 0.8)
    = 0.06 + 0.05 + 0.32
    = 0.43

y = f(0.43 + 0.1) = f(0.53)
```

### Fonctions d'Activation

**C'est quoi ?** Des fonctions mathématiques qui ajoutent de la **non-linéarité**.

**Pourquoi ?** Sans elles, le réseau serait juste une grosse addition (= trop simple).

#### 1. **ReLU** (Rectified Linear Unit) - LA PLUS UTILISÉE

```
f(x) = max(0, x)
```

**En français** : "Si x est positif, garde-le. Sinon, mets 0."

Exemple :
- `f(5) = 5`
- `f(-3) = 0`
- `f(0.7) = 0.7`

**Graphique** :
```
 f(x)
  |
  |     /
  |   /
  | /
  |__________ x
       0
```

**Pourquoi ReLU ?**
- ✅ Super simple (= rapide)
- ✅ Marche très bien en pratique
- ✅ Évite le "gradient vanishing" (problème technique)

#### 2. **Sigmoid** - Pour les probabilités

```
f(x) = 1 / (1 + e^(-x))
```

**En français** : "Transforme n'importe quel nombre en probabilité entre 0 et 1"

Exemple :
- `f(-5) ≈ 0.007` (proche de 0)
- `f(0) = 0.5` (50%)
- `f(5) ≈ 0.993` (proche de 1)

**Graphique** :
```
 f(x)
  1 |     ___________
    |   /
0.5 | /
    |/_______________x
  0
```

**Utilisation** : Couche de sortie pour la **classification binaire** (Normal vs Pneumonie)

#### 3. **Softmax** - Pour plusieurs classes

```
f(x_i) = e^(x_i) / Σ(e^(x_j))
```

**En français** : "Transforme plusieurs nombres en probabilités qui somment à 1"

Exemple :

Input : `[2.0, 1.0, 0.1]`

```
e^2.0 = 7.39
e^1.0 = 2.72
e^0.1 = 1.11
Somme = 11.22

Softmax:
  Normal   = 7.39 / 11.22 = 0.66 (66%)
  Bactérie = 2.72 / 11.22 = 0.24 (24%)
  Virus    = 1.11 / 11.22 = 0.10 (10%)
```

**Utilisation** : Couche de sortie pour la **classification multi-classes**

---

### L'Entraînement : Comment l'IA Apprend

**Problème** : Au début, les poids `W` sont **aléatoires** → Le réseau dit n'importe quoi

**Solution** : L'**entraînement** = ajuster les poids pour que le réseau s'améliore

#### Les 4 Étapes de l'Entraînement

**1. Forward Pass** (Passage avant)
- On donne une image au réseau
- Il fait sa prédiction
- Ex: "Je pense que c'est Normal à 80%"

**2. Calcul de l'Erreur (Loss)**
- On compare la prédiction à la vraie réponse
- On calcule **à quel point il s'est trompé**

**Fonction de Loss** la plus courante : **Cross-Entropy**

```
Loss = -Σ y_true * log(y_pred)
```

Exemple :

Vraie classe : `Pneumonie = [0, 1, 0]` (one-hot encoding)

Prédiction : `[0.8, 0.15, 0.05]` (le modèle pense Normal)

```
Loss = -(0×log(0.8) + 1×log(0.15) + 0×log(0.05))
     = -log(0.15)
     = 1.90
```

**Plus la loss est élevée, plus le modèle s'est trompé.**

**3. Backpropagation** (Rétropropagation)
- On calcule le **gradient** (= dans quelle direction ajuster les poids)
- C'est de la **dérivée en chaîne** (maths niveau Terminale/Bac+1)

**Intuition** :
> "Si j'augmente ce poids de 0.01, est-ce que la loss diminue ou augmente ?"

Le gradient dit : "Augmente ce poids" ou "Diminue ce poids"

**4. Mise à Jour des Poids (Optimization)**

```
W_nouveau = W_ancien - learning_rate × gradient
```

- `learning_rate` = Taille du pas (souvent 0.001 ou 0.0001)
- On ajuste TOUS les poids du réseau

**Exemple** :

Poids actuel : `W = 0.5`

Gradient : `∇L = 0.3` (la loss augmente si on augmente W)

Learning rate : `α = 0.01`

```
W_nouveau = 0.5 - 0.01 × 0.3 = 0.497
```

On a **baissé** W pour réduire la loss.

### Epochs et Batch

**Epoch** = 1 passage complet sur TOUTES les données

Si tu as 1000 images :
- 1 epoch = le modèle voit les 1000 images une fois

Tu entraînes souvent **10 à 50 epochs**

**Batch** = Groupe d'images traitées ensemble

Au lieu de faire :
- 1 image → ajuster poids
- 1 image → ajuster poids
- ...

On fait :
- 32 images → moyenne des gradients → ajuster poids
- 32 images → moyenne → ajuster
- ...

**Batch size** classique : 32, 64, 128

**Pourquoi ?**
- Plus rapide (calcul parallèle sur GPU)
- Plus stable (moyenne des gradients)

---

# 3. CNN : Comment ça Marche ?

## 🖼️ Le Problème avec les Images

**Image 128×128 pixels RGB** = 128 × 128 × 3 = **49,152 valeurs**

Si on met tout ça dans un réseau classique :
- Il faut **49,152 poids** rien que pour la première couche
- Le réseau ne comprend PAS la **structure spatiale** (un œil est à côté d'un nez, etc.)

**Solution** : Les **CNN** (Convolutional Neural Networks)

## 🔍 Convolution : L'Idée Géniale

### Analogie : Le Tampon Encreur

Imagine un **petit tampon** (3×3 pixels) que tu promènes sur l'image.

À chaque endroit :
1. Tu **multiplies** les pixels par le tampon
2. Tu **additionnes** tout
3. Tu obtiens **un nombre**
4. Tu passes au carré suivant

**Ce tampon = un Filtre (ou Kernel)**

### Exemple Concret

**Image (5×5)** :
```
[0 0 1 0 0]
[0 0 1 0 0]
[0 0 1 0 0]
[0 0 1 0 0]
[0 0 1 0 0]
```
(Une ligne verticale)

**Filtre (3×3)** pour détecter les lignes verticales :
```
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

**Calcul** en position (1,1) :

```
[0  0  1]       [-1  0  1]
[0  0  1]   ×   [-1  0  1]  
[0  0  1]       [-1  0  1]

= 0×(-1) + 0×0 + 1×1 +
  0×(-1) + 0×0 + 1×1 +
  0×(-1) + 0×0 + 1×1
  
= 0 + 0 + 1 + 0 + 0 + 1 + 0 + 0 + 1
= 3
```

**Résultat** : Le filtre a **activé** (valeur élevée) car il a détecté une ligne verticale !

### Ce Que Font les Filtres

Chaque filtre détecte un **motif** :
- Filtre 1 : Lignes horizontales
- Filtre 2 : Lignes verticales
- Filtre 3 : Coins
- Filtre 4 : Courbes
- ...

**Avec 32 filtres, tu détectes 32 motifs différents !**

### Architecture CNN Complète

```
INPUT (128×128×3)
    ↓
[Conv 32 filtres] → Détecte 32 motifs de base
    ↓
[ReLU] → Activation
    ↓
[MaxPooling 2×2] → Réduit la taille (→ 64×64)
    ↓
[Conv 64 filtres] → Détecte des motifs plus complexes
    ↓
[ReLU]
    ↓
[MaxPooling 2×2] → 32×32
    ↓
[Conv 128 filtres] → Détecte des objets entiers
    ↓
[Flatten] → Transforme en vecteur 1D
    ↓
[Dense 128 neurones] → Classification
    ↓
[Dropout] → Régularisation (évite surapprentissage)
    ↓
[Dense 3 neurones] → Sortie (Normal, Bactérie, Virus)
    ↓
[Softmax] → Probabilités
```

### Max Pooling : Réduction de Dimension

**Idée** : "On garde juste le maximum dans une zone"

**Exemple** avec un pool 2×2 :

Input (4×4) :
```
[1  3  2  4]
[5  6  1  2]
[3  1  0  8]
[4  2  1  7]
```

On découpe en blocs 2×2 et on garde le max :

```
Bloc 1:        Bloc 2:
[1  3]         [2  4]
[5  6]  →  6   [1  2]  →  4

Bloc 3:        Bloc 4:
[3  1]         [0  8]
[4  2]  →  4   [1  7]  →  8
```

Output (2×2) :
```
[6  4]
[4  8]
```

**Pourquoi ?**
- ✅ Réduit la taille (moins de calculs)
- ✅ Garde l'info importante (les activations fortes)
- ✅ Rend le réseau **invariant aux petits décalages**

---

# 4. Transfer Learning Expliqué

## 🚀 L'Idée de Génie

**Problème** : Entraîner un CNN from scratch demande :
- Des **millions** d'images
- Des **jours/semaines** d'entraînement
- Des **GPU très puissants**

**Solution** : Le **Transfer Learning**

> "Utiliser un modèle déjà entraîné par Google/Facebook sur des millions d'images, et l'adapter à TON problème"

## 🎓 Analogie : Le Médecin Spécialiste

**Médecin généraliste (= Modèle pré-entraîné)**
- A fait 8 ans d'études de médecine générale
- Connaît l'anatomie, la biologie, les pathologies

**Devient radiologue (= Fine-tuning)**
- Pas besoin de refaire les 8 ans !
- Juste **2-3 ans de spécialisation** en radiologie
- Il **adapte** ses connaissances générales

**Transfer Learning = PAREIL**

## 📚 Les Modèles Pré-Entraînés

### ImageNet : Le Dataset de Référence

- **14 millions** d'images
- **1000 classes** (chat, chien, voiture, fleur, etc.)
- Google/Facebook ont entraîné des modèles dessus pendant des semaines

### EfficientNet : Le Modèle Que Tu Utilises

**EfficientNetB0** = Version la plus petite (mais déjà très bonne)

Architecture optimisée pour :
- ✅ Être **rapide**
- ✅ Utiliser **peu de mémoire**
- ✅ Avoir de **bonnes performances**

**Nombre de paramètres** : ~5 millions (contre ~60 millions pour ResNet)

**Top-1 Accuracy sur ImageNet** : 77.1%

## 🔧 Comment Tu l'Utilises

### Étape 1 : Charger le Modèle Pré-Entraîné

```python
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    weights='imagenet',  # Poids pré-entraînés sur ImageNet
    include_top=False,    # On enlève la dernière couche
    input_shape=(128, 128, 3)
)
```

**`include_top=False`** = On enlève la couche de classification (1000 classes ImageNet)

Tu gardes juste les **couches de convolution** (= détecteurs de motifs)

### Étape 2 : Geler les Couches

```python
# Geler toutes les couches
for layer in base_model.layers:
    layer.trainable = False

# Dégeler les 20 dernières couches
for layer in base_model.layers[-20:]:
    layer.trainable = True
```

**Pourquoi geler ?**
- Les premières couches détectent des **motifs basiques** (bords, textures)
- Ces motifs sont **universels** (utiles pour toutes les images)
- Pas besoin de les réentraîner !

**Pourquoi dégeler les 20 dernières ?**
- Les dernières couches détectent des **motifs complexes**
- Elles doivent s'adapter aux **radios médicales** (différentes de ImageNet)

### Étape 3 : Ajouter une Nouvelle Tête de Classification

```python
from tensorflow.keras import layers, models

# Construire le modèle complet
model = models.Sequential([
    base_model,                              # Couches EfficientNet gelées
    layers.GlobalAveragePooling2D(),         # Réduit à un vecteur
    layers.BatchNormalization(),             # Normalisation
    layers.Dropout(0.5),                     # Régularisation
    layers.Dense(128, activation='relu'),    # Couche cachée
    layers.Dropout(0.3),                     # Encore de la régularisation
    layers.Dense(3, activation='softmax')    # 3 classes finales
])
```

### Étape 4 : Entraîner avec un Petit Learning Rate

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Très petit !
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=20)
```

**Learning rate = 1e-5 = 0.00001** (très petit !)

**Pourquoi si petit ?**
- Les poids pré-entraînés sont **déjà bons**
- On veut juste les **ajuster légèrement**
- Pas les **casser** avec de gros changements

---

# 5. Ton Pipeline Hiérarchique

## 🎯 Le Problème

Tu as essayé un modèle direct à 3 classes :

```
Image → [Modèle] → Normal / Bactérie / Virus
```

**Résultat** : 56% accuracy (pas terrible)

**Pourquoi ?**
- Le modèle doit apprendre **2 tâches en même temps** :
  1. Distinguer sain vs malade (facile)
  2. Distinguer bactérie vs virus (difficile)
- Il se **mélange les pinceaux**

## 💡 Ton Innovation : Diviser Pour Mieux Régner

**Idée** : Créer **2 modèles spécialisés**

```
Image → [Modèle 1 : Binaire] → Normal ?
                             → Pneumonie ? → [Modèle 2 : Sous-type] → Bactérie ?
                                                                     → Virus ?
```

**Résultat** : 71% accuracy (+15 points !)

## 🔧 Les 2 Modèles

### Modèle 1 : EfficientNet Binaire (Stage 1)

**Tâche** : Normal vs Pneumonie (toutes confondues)

**Architecture** :
```python
# Base EfficientNet
base = EfficientNetB0(weights='imagenet', include_top=False)

# Geler les couches
for layer in base.layers[:-20]:
    layer.trainable = False

# Nouvelle tête binaire
model = Sequential([
    base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # 1 neurone = binaire
])
```

**Activation finale** : Sigmoid (probabilité entre 0 et 1)
- `< 0.5` → Normal
- `≥ 0.5` → Pneumonie

**Performance** :
- AUC : **0.9897** (quasi-parfait !)
- Recall : **98%** (ne rate presque aucun malade)

### Modèle 2 : Expert Sous-type (Stage 2)

**Tâche** : Bactérie vs Virus (uniquement sur les pneumonies)

**Architecture** : Identique au Stage 1, mais entraîné UNIQUEMENT sur les images de pneumonie

**Performance** :
- Accuracy : **76%**
- Recall Bactérie : **98%** (excellent !)
- Recall Virus : **48%** (plus difficile)

## 🔄 Le Flux Complet

**Exemple 1** : Radio d'un poumon sain

```
Radio → Stage 1 : "Normal (95%)" → FIN
```

Résultat : **Normal** ✅

**Exemple 2** : Radio d'une pneumonie bactérienne

```
Radio → Stage 1 : "Pneumonie (98%)" 
           ↓
      Stage 2 : "Bactérie (92%)"
```

Résultat : **Bactérie** ✅

**Exemple 3** : Radio d'une pneumonie virale

```
Radio → Stage 1 : "Pneumonie (85%)"
           ↓
      Stage 2 : "Virus (62%)"
```

Résultat : **Virus** ✅

## 📊 Pourquoi Ça Marche Mieux ?

**1. Spécialisation**
- Chaque modèle est **expert** sur sa tâche
- Pas de confusion entre les problèmes

**2. Imite le Raisonnement Médical**
- Un radiologue fait PAREIL :
  1. "Y a-t-il une anomalie ?" (Stage 1)
  2. "Si oui, quelle est sa nature ?" (Stage 2)

**3. Les Erreurs sont Localisées**
- Si Stage 1 se trompe → erreur sur Normal/Pneumonie
- Si Stage 2 se trompe → erreur sur Bactérie/Virus
- Mais **pas les deux en même temps** !

---

# 6. Les Métriques Médicales

## 📊 La Matrice de Confusion

**Tableau qui montre toutes les prédictions :**

```
                    Prédiction
                 Normal  Pneumonie
    Vrai  Normal    TP        FP
          Pneumonie FN        TP
```

- **TP** (True Positive) : J'ai dit malade, c'est vrai
- **FP** (False Positive) : J'ai dit malade, c'est faux (fausse alerte)
- **FN** (False Négatif) : J'ai dit sain, mais c'est faux (DANGER !)
- **TN** (True Negative) : J'ai dit sain, c'est vrai

## 🎯 Les Métriques Clés

### 1. Accuracy (Précision Globale)

```
Accuracy = (TP + TN) / Total
```

**En français** : "% de bonnes prédictions"

**Exemple** :
- 100 images
- 85 bien classées

```
Accuracy = 85 / 100 = 85%
```

**⚠️ Problème** : Pas adaptée si dataset déséquilibré

Si tu as :
- 95 Normal
- 5 Pneumonie

Un modèle qui dit **TOUJOURS Normal** a 95% accuracy... mais rate TOUS les malades !

### 2. Precision (Précision)

```
Precision = TP / (TP + FP)
```

**En français** : "Quand je dis malade, est-ce que j'ai raison ?"

**Exemple** :
- J'ai détecté 100 pneumonies
- 85 sont vraiment des pneumonies
- 15 sont en fait normales (fausses alertes)

```
Precision = 85 / 100 = 85%
```

**Interprétation** : "Quand je crie au loup, c'est vrai 85% du temps"

### 3. Recall (Sensibilité / Rappel)

```
Recall = TP / (TP + FN)
```

**En français** : "Sur tous les vrais malades, combien j'en détecte ?"

**Exemple** :
- Il y a 100 pneumonies dans le dataset
- J'en détecte 98
- J'en rate 2

```
Recall = 98 / 100 = 98%
```

**Interprétation** : "Je ne rate que 2% des malades"

**🏥 EN MÉDICAL, LE RECALL EST PRIORITAIRE !**

Pourquoi ?
- **Faux Positif** (FP) : Le patient fait des tests supplémentaires pour rien → Pas grave
- **Faux Négatif** (FN) : Le patient malade est renvoyé chez lui → **DANGER !**

**Ton modèle a 98% Recall** = Il ne rate presque aucun malade ✅

### 4. F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**En français** : "Moyenne harmonique de Precision et Recall"

C'est un **compromis** entre les deux.

**Exemple** :
- Precision = 80%
- Recall = 90%

```
F1 = 2 × (0.80 × 0.90) / (0.80 + 0.90)
   = 2 × 0.72 / 1.70
   = 0.847
   = 84.7%
```

### 5. AUC-ROC (Area Under Curve)

**C'est quoi ?**

La **courbe ROC** trace :
- X-axis : False Positive Rate (taux de faux positifs)
- Y-axis : True Positive Rate (= Recall)

On fait varier le **seuil de décision** :
- Seuil 0.1 → Presque tout est "Pneumonie"
- Seuil 0.9 → Presque tout est "Normal"

**L'AUC** = Aire sous cette courbe

```
AUC = 0.5  → Modèle aléatoire (lance une pièce)
AUC = 0.7  → Modèle acceptable
AUC = 0.8  → Bon modèle
AUC = 0.9  → Très bon modèle
AUC = 1.0  → Modèle parfait
```

**Ton modèle : AUC = 0.9897** → **Quasi-parfait !** 🔥

---

# 7. Grad-CAM et Explicabilité

## 👁️ Le Problème

**Boîte noire** : Le CNN dit "C'est une pneumonie" mais on ne sait pas **POURQUOI**

Le médecin ne peut pas **faire confiance** à une boîte noire.

## 💡 La Solution : Grad-CAM

**Grad-CAM** = Gradient-weighted Class Activation Mapping

**Idée** : "Montrer quelles zones de l'image ont activé la décision"

### Comment Ça Marche (Simplifié)

**Étape 1** : Le modèle fait sa prédiction
```
Image → CNN → "Pneumonie (85%)"
```

**Étape 2** : On regarde la **dernière couche de convolution**

Cette couche contient les **feature maps** (cartes de caractéristiques)

Ex: 64 feature maps de taille 16×16

**Étape 3** : On calcule le **gradient**

> "Si je modifie cette feature map, comment la prédiction change ?"

Feature map qui influence beaucoup → Gradient élevé

**Étape 4** : On pondère les feature maps par leur importance

```
Heatmap = Σ (gradient_i × feature_map_i)
```

**Étape 5** : On superpose la heatmap sur l'image originale

**Résultat** : Zones **rouges** = Zones que le modèle a **regardées**

### Interprétation

**Bonne prédiction** :
- Zones rouges sur les **poumons** ✅
- Le modèle regarde les **opacités** pulmonaires ✅

**Mauvaise prédiction (Shortcut Learning)** :
- Zones rouges sur les **lettres "L/R"** ❌
- Zones rouges sur les **bords métalliques** ❌
- Le modèle **triche** en regardant des artefacts ❌

### Ton Utilisation

Tu as implémenté Grad-CAM dans `grad_cam.py` pour :
1. **Valider** que le modèle regarde les bons endroits
2. **Expliquer** les erreurs (pourquoi il s'est trompé)
3. **Rassurer** le médecin (preuve visuelle)

---

# 8. Résumé pour l'Oral

## 🎤 Pitch de 2 Minutes

> "Mon projet Zoidberg utilise le Deep Learning pour aider les radiologues à détecter la pneumonie sur des radios thoraciques.
>
> J'ai comparé 5 architectures : une baseline machine learning classique, un CNN from scratch, du transfer learning avec EfficientNet, et finalement un **pipeline hiérarchique à 2 étages** qui est mon innovation principale.
>
> L'idée : diviser le problème complexe en 2 sous-problèmes plus simples. Le premier modèle détecte si le patient est sain ou malade (98% de recall, AUC de 0.99). Si malade, le second modèle spécialisé détermine si c'est viral ou bactérien (76% accuracy).
>
> Cette approche hiérarchique améliore les performances de +15 points par rapport à un modèle multi-classes direct (71% vs 56%).
>
> Pour garantir la confiance médicale, j'ai intégré Grad-CAM, une technique d'explicabilité qui montre visuellement où le modèle regarde dans l'image pour prendre sa décision. Ça permet au radiologue de valider que l'IA analyse bien les poumons et pas des artefacts.
>
> Le système atteint un recall de 98% sur la détection de pneumonie, ce qui est crucial en médical : on préfère une fausse alerte qu'un patient malade renvoyé chez lui."

## 🧠 Points Techniques à Maîtriser

### Si on te demande "C'est quoi un CNN ?"

> "Un CNN, c'est un réseau de neurones spécialisé pour les images. Au lieu d'analyser tous les pixels d'un coup, il utilise des filtres de convolution qui parcourent l'image pour détecter des motifs : d'abord des bords et textures, puis des formes plus complexes, et enfin des objets entiers. Chaque couche de convolution extrait des caractéristiques de plus en plus abstraites."

### Si on te demande "C'est quoi le Transfer Learning ?"

> "C'est réutiliser un modèle déjà entraîné sur des millions d'images (ImageNet) et l'adapter à mon problème spécifique. Comme un médecin généraliste qui se spécialise en radiologie : il garde ses connaissances de base et ajoute une spécialisation. J'ai utilisé EfficientNet pré-entraîné, gelé les premières couches qui détectent des motifs universels, et fine-tuné les 20 dernières couches sur mes radios médicales."

### Si on te demande "Pourquoi hiérarchique ?"

> "J'ai d'abord essayé un modèle direct à 3 classes : 56% accuracy. Le modèle devait apprendre 2 tâches différentes en même temps et se mélangeait. J'ai donc divisé le problème : un modèle spécialiste détecte sain vs malade (quasi-parfait avec 98% recall), puis si malade, un second modèle expert différencie bactérie vs virus. Chaque modèle devient expert sur SA tâche. Résultat : 71% accuracy, +15 points. C'est comme le raisonnement médical réel : d'abord détecter l'anomalie, ensuite la qualifier."

### Si on te demande "C'est quoi Grad-CAM ?"

> "Grad-CAM génère une carte de chaleur qui montre quelles zones de l'image ont influencé la décision du modèle. Techniquement, on calcule le gradient de la prédiction par rapport à la dernière couche de convolution, ce qui donne l'importance de chaque zone. On superpose cette heatmap sur l'image : zones rouges = zones regardées par le modèle. Ça permet de vérifier qu'il analyse bien les poumons et pas des artefacts, et d'expliquer ses erreurs."

### Si on te demande "Pourquoi 98% Recall ?"

> "En médical, le Recall est prioritaire sur l'Accuracy. Un faux positif (dire malade alors que sain), c'est désagréable pour le patient mais pas dangereux : on fait des tests supplémentaires. Un faux négatif (dire sain alors que malade), c'est grave : le patient malade rentre chez lui. Mon modèle atteint 98% Recall : il ne rate que 2% des malades. C'est ce qu'on veut pour un outil d'aide au diagnostic."

## 🔑 Cheat Sheet : Vocabulaire Clé

| Terme | Explication Simple | Formule |
|-------|-------------------|---------|
| **Neurone** | Unité de calcul : multiplie les inputs par des poids, ajoute un biais, applique une fonction | `y = f(W·x + b)` |
| **ReLU** | Fonction d'activation : garde les positifs, met les négatifs à 0 | `f(x) = max(0, x)` |
| **Convolution** | Filtre qui parcourt l'image pour détecter des motifs | Produit matriciel local |
| **MaxPooling** | Réduit la taille en gardant le maximum dans chaque zone | Max dans une fenêtre |
| **Dropout** | Éteint aléatoirement des neurones pour éviter le surapprentissage | - |
| **Sigmoid** | Transforme en probabilité entre 0 et 1 | `1/(1+e^(-x))` |
| **Softmax** | Transforme N scores en N probabilités qui somment à 1 | `e^xi / Σe^xj` |
| **Loss** | Mesure de l'erreur du modèle | Cross-Entropy |
| **Backprop** | Calcul des gradients pour ajuster les poids | Dérivée en chaîne |
| **Epoch** | 1 passage complet sur toutes les données | - |
| **Batch** | Groupe d'images traitées ensemble | - |
| **Learning Rate** | Taille du pas d'ajustement des poids | Souvent 0.001 |
| **Accuracy** | % de bonnes prédictions | `(TP+TN)/Total` |
| **Precision** | Quand je dis positif, ai-je raison ? | `TP/(TP+FP)` |
| **Recall** | Sur tous les positifs, combien j'en trouve ? | `TP/(TP+FN)` |
| **F1-Score** | Moyenne harmonique Precision/Recall | `2PR/(P+R)` |
| **AUC** | Aire sous la courbe ROC (0.5=aléatoire, 1=parfait) | Intégrale |

## 📚 Si On Creuse les Maths

**Backpropagation - Exemple Simplifié**

Réseau simple : `x → [W] → y → [Loss]`

```
Forward:
  y = W × x = 2 × 3 = 6
  Loss = (y - target)² = (6 - 5)² = 1
  
Backward (dérivée) :
  ∂Loss/∂y = 2(y - target) = 2(6 - 5) = 2
  ∂y/∂W = x = 3
  ∂Loss/∂W = (∂Loss/∂y) × (∂y/∂W) = 2 × 3 = 6
  
Update (learning_rate = 0.01) :
  W_new = W - α × ∂Loss/∂W
        = 2 - 0.01 × 6
        = 1.94
```

Le gradient dit : "La loss augmente si W augmente" → On **baisse** W

---

## ✅ Checklist Oral

Avant ton oral, assure-toi de pouvoir expliquer :

- [ ] Le problème médical et pourquoi c'est important
- [ ] C'est quoi un réseau de neurones (analogie + formule de base)
- [ ] C'est quoi un CNN et pourquoi c'est adapté aux images
- [ ] C'est quoi le Transfer Learning et pourquoi tu l'utilises
- [ ] Pourquoi tu as choisi une approche hiérarchique
- [ ] Les 2 modèles de ton pipeline et leur rôle
- [ ] Les métriques clés (Accuracy, Recall, AUC) et pourquoi Recall est prioritaire
- [ ] C'est quoi Grad-CAM et pourquoi c'est important en médical
- [ ] Tes résultats chiffrés (71% accuracy, 98% recall, AUC 0.99)
- [ ] Les limites du projet (distinction virus/bactérie difficile)

---

**🎉 TU ES PRÊT POUR TON ORAL MON BRO !**

*Document créé le 10/06/2026 - Guide pédagogique complet Zoidberg*
