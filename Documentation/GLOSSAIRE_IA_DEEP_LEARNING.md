# 📖 GLOSSAIRE - Intelligence Artificielle & Deep Learning

**Projet** : Zoidberg - Détection de Pneumonie  
**Date** : Juin 2026  
**Public** : Débutants en IA / Étudiants / Jury d'évaluation

---

## 🎯 Comment utiliser ce glossaire

- **Format** : Terme = Définition + Analogie + Contexte Zoidberg
- **Niveau** : Vulgarisé mais précis
- **Organisation** : Par catégories (Architecture, Entraînement, Évaluation, Médical)

---

## 📚 CATÉGORIES

1. [Architectures de Réseaux de Neurones](#1-architectures-de-réseaux-de-neurones)
2. [Composants des CNN](#2-composants-des-cnn)
3. [Entraînement & Optimisation](#3-entraînement--optimisation)
4. [Métriques d'Évaluation](#4-métriques-dévaluation)
5. [Métriques Médicales Spécifiques](#5-métriques-médicales-spécifiques)
6. [Techniques Avancées](#6-techniques-avancées)
7. [Transfer Learning & Fine-Tuning](#7-transfer-learning--fine-tuning)
8. [Validation & Tests](#8-validation--tests)
9. [Explicabilité (XAI)](#9-explicabilité-xai)
10. [Termes Médicaux](#10-termes-médicaux)

---

## 1. Architectures de Réseaux de Neurones

### CNN (Convolutional Neural Network)
**Définition** : Réseau de neurones spécialisé pour traiter des images.  
**Analogie** : Comme un tampon encreur qui glisse sur une image pour détecter des motifs (bords, textures, formes).  
**Dans Zoidberg** : Notre "CNN from scratch" (v1) analyse les radiographies pixel par pixel.

### EfficientNet
**Définition** : Architecture CNN moderne optimisée pour être légère et performante.  
**Analogie** : Un expert médical qui a déjà vu des millions d'images (pré-entraîné sur ImageNet).  
**Dans Zoidberg** : EfficientNetB0 utilisé en Transfer Learning pour détecter les pneumonies (AUC 0.9897).

### ResNet (Residual Network)
**Définition** : Architecture avec connexions résiduelles permettant d'entraîner des réseaux très profonds.  
**Analogie** : Des raccourcis dans un labyrinthe qui évitent de se perdre.  
**Dans Zoidberg** : Non utilisé, mais alternative à EfficientNet.

### Pipeline Hiérarchique
**Définition** : Système en 2 étapes : d'abord détecter SI malade, puis QUEL type.  
**Analogie** : Médecin généraliste (Stage 1) → Spécialiste (Stage 2).  
**Dans Zoidberg** : **Notre innovation principale** (+15% accuracy vs approche directe).

---

## 2. Composants des CNN

### Convolution
**Définition** : Opération mathématique qui applique un filtre sur l'image pour détecter des caractéristiques.  
**Analogie** : Passer une loupe sur une photo pour chercher des détails spécifiques.  
**Formule** : `Output = Input ⊗ Kernel + Bias`  
**Dans Zoidberg** : Détecte les opacités pulmonaires dans les radiographies.

### Pooling (Max Pooling)
**Définition** : Réduit la taille de l'image en gardant seulement les valeurs maximales.  
**Analogie** : Zoomer en arrière pour voir la vue d'ensemble.  
**Rôle** : Diminue le nombre de paramètres et rend le modèle plus robuste.  
**Dans Zoidberg** : Utilisé après chaque couche de convolution.

### Dropout
**Définition** : Désactive aléatoirement des neurones pendant l'entraînement pour éviter le surapprentissage.  
**Analogie** : Faire réviser un étudiant en lui enlevant aléatoirement des fiches → force à généraliser.  
**Valeur typique** : 0.2 à 0.5 (20-50% de neurones désactivés).  
**Dans Zoidberg** : Dropout(0.5) utilisé pour éviter l'overfitting.

### Batch Normalization
**Définition** : Normalise les activations entre couches pour stabiliser l'entraînement.  
**Analogie** : Standardiser les notes d'un examen pour que toutes les matières aient la même échelle.  
**Bénéfice** : Permet d'entraîner plus vite et plus profond.

### Flatten
**Définition** : Transforme une matrice 2D en vecteur 1D.  
**Analogie** : Dérouler un tapis pour en faire une ligne.  
**Rôle** : Connecte les couches convolutionnelles aux couches denses (fully connected).

### Dense (Fully Connected)
**Définition** : Couche où chaque neurone est connecté à tous les neurones de la couche précédente.  
**Analogie** : Réunion où tout le monde parle à tout le monde.  
**Dans Zoidberg** : Couche finale qui fait la classification (3 neurones pour 3 classes).

---

## 3. Entraînement & Optimisation

### Epoch
**Définition** : Une passe complète sur tout le dataset d'entraînement.  
**Analogie** : Relire un livre entier une fois.  
**Dans Zoidberg** : On entraîne généralement 20-50 epochs.

### Batch Size
**Définition** : Nombre d'images traitées simultanément avant de mettre à jour les poids.  
**Analogie** : Corriger 32 copies avant de noter les erreurs.  
**Dans Zoidberg** : Batch size = 32 (compromis vitesse/mémoire).

### Learning Rate
**Définition** : Taille des pas que fait le modèle pour ajuster ses poids.  
**Analogie** : Vitesse à laquelle on descend une montagne (trop vite = on tombe, trop lent = on n'avance pas).  
**Valeur typique** : 0.001 à 0.0001.  
**Dans Zoidberg** : Learning rate adaptatif avec Adam optimizer.

### Loss Function (Fonction de Perte)
**Définition** : Mesure l'erreur entre prédictions et vérité terrain.  
**Types** :
- **Binary Crossentropy** : Pour classification binaire (Normal vs Pneumonie).
- **Categorical Crossentropy** : Pour classification multi-classes (Normal/Bactérie/Virus).
- **Focal Loss** : Pénalise plus les erreurs sur classes difficiles.  
**Dans Zoidberg** : Categorical Crossentropy utilisé.

### Optimizer
**Définition** : Algorithme qui ajuste les poids du réseau pour minimiser la loss.  
**Types** :
- **SGD** : Descente de gradient simple.
- **Adam** : Adaptatif, combine momentum et RMSprop (le plus populaire).  
**Dans Zoidberg** : Adam optimizer (learning rate auto-ajusté).

### Backpropagation
**Définition** : Mécanisme de calcul des gradients pour mettre à jour les poids.  
**Analogie** : Remonter la chaîne de responsabilité pour trouver qui a fait l'erreur.  
**Formule** : Dérivée en chaîne pour propager l'erreur de la sortie vers l'entrée.

### Gradient Descent
**Définition** : Algorithme d'optimisation qui descend le gradient de la fonction de perte.  
**Analogie** : Descendre une montagne en suivant la pente la plus raide.  
**Types** : SGD, Mini-batch SGD, Adam.

---

## 4. Métriques d'Évaluation

### Accuracy (Exactitude)
**Définition** : Pourcentage de prédictions correctes.  
**Formule** : `Accuracy = (TP + TN) / Total`  
**Limite** : Trompeuse sur datasets déséquilibrés.  
**Dans Zoidberg** : Pipeline = 71% accuracy (meilleur global).

### Precision (Précision)
**Définition** : Proportion de prédictions positives correctes.  
**Formule** : `Precision = TP / (TP + FP)`  
**Question** : "Quand le modèle dit 'Pneumonie', combien de fois a-t-il raison ?"  
**Importance** : Évite les fausses alertes.

### Recall (Sensibilité / Sensitivity)
**Définition** : Proportion de vrais positifs détectés.  
**Formule** : `Recall = TP / (TP + FN)`  
**Question** : "Parmi tous les malades, combien le modèle en détecte-t-il ?"  
**Dans Zoidberg** : **MÉTRIQUE PRIORITAIRE** (Recall = 98% sur pneumonies). On ne veut RATER AUCUN MALADE.

### F1-Score
**Définition** : Moyenne harmonique de Precision et Recall.  
**Formule** : `F1 = 2 × (Precision × Recall) / (Precision + Recall)`  
**Rôle** : Équilibre entre Precision et Recall.  
**Dans Zoidberg** : F1-Score Macro = 65% (Pipeline).

### AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)
**Définition** : Mesure la capacité du modèle à discriminer entre classes sur tous les seuils possibles.  
**Interprétation** :
- 0.5 = Aléatoire
- 0.7-0.8 = Acceptable
- 0.8-0.9 = Excellent
- 0.9+ = Exceptionnel  
**Dans Zoidberg** : AUC = 0.9897 (quasi-parfait pour détection binaire).

### Confusion Matrix (Matrice de Confusion)
**Définition** : Tableau montrant les prédictions vs vérité terrain.  
**Format** :
```
              Predicted
              N    B    V
Actual   N  [TN   FP1  FP2]
         B  [FN1  TP   FP3]
         V  [FN2  FN3  TP ]
```
**Dans Zoidberg** : Montre que Virus/Bactérie sont souvent confondus (limite médicale).

---

## 5. Métriques Médicales Spécifiques

### Specificity (Spécificité)
**Définition** : Proportion de vrais négatifs correctement identifiés.  
**Formule** : `Specificity = TN / (TN + FP)`  
**Question** : "Parmi les personnes saines, combien sont correctement identifiées ?"  
**Importance Médicale** : Évite de stresser inutilement des patients sains.

### NPV (Negative Predictive Value / Valeur Prédictive Négative)
**Définition** : Probabilité qu'un patient testé négatif soit vraiment sain.  
**Formule** : `NPV = TN / (TN + FN)`  
**Question** : "Si le modèle dit 'Normal', quelle confiance avoir ?"  
**Utilité** : Rassurer les patients avec résultat négatif.

### PPV (Positive Predictive Value / Valeur Prédictive Positive)
**Définition** : Probabilité qu'un patient testé positif soit vraiment malade.  
**Formule** : `PPV = TP / (TP + FP)` = Precision  
**Question** : "Si le modèle dit 'Pneumonie', quelle confiance avoir ?"  
**Utilité** : Décider si traitement immédiat ou tests complémentaires.

### Cohen's Kappa
**Définition** : Mesure l'accord entre prédictions et vérité en tenant compte du hasard.  
**Interprétation** :
- < 0 : Pire que le hasard
- 0-0.2 : Accord faible
- 0.2-0.4 : Accord modéré
- 0.4-0.6 : Accord bon
- 0.6-0.8 : Accord fort
- 0.8-1.0 : Accord quasi-parfait  
**Utilité** : Évaluer la robustesse sur datasets déséquilibrés.

### MCC (Matthews Correlation Coefficient)
**Définition** : Corrélation entre prédictions et vérité (-1 à +1).  
**Avantage** : Fonctionne bien même si classes très déséquilibrées.  
**Interprétation** :
- +1 = Prédiction parfaite
- 0 = Pas mieux que le hasard
- -1 = Désaccord total

### Balanced Accuracy
**Définition** : Moyenne des Recall de chaque classe.  
**Formule** : `Balanced Acc = (Recall_Class1 + Recall_Class2 + ...) / N_Classes`  
**Avantage** : Corrige le biais des datasets déséquilibrés.  
**Dans Zoidberg** : Plus fiable que Accuracy standard (47% Bactéries, 27% Normal, 26% Virus).

---

## 6. Techniques Avancées

### Data Augmentation
**Définition** : Créer des variations artificielles des images d'entraînement.  
**Techniques** :
- Rotation (±20°)
- Zoom (±20%)
- Flip horizontal
- Brightness (luminosité ±20%)  
**But** : Augmenter artificiellement la taille du dataset et la robustesse.  
**Dans Zoidberg** : Utilisé avec `ImageDataGenerator` de Keras.

### Overfitting (Surapprentissage)
**Définition** : Le modèle "apprend par cœur" les données d'entraînement mais ne généralise pas.  
**Symptôme** : Accuracy Train >> Accuracy Validation.  
**Solutions** : Dropout, Regularization, Early Stopping, Data Augmentation.  
**Dans Zoidberg** : Gap Train-Val < 5% → Pas d'overfitting significatif.

### Underfitting (Sous-apprentissage)
**Définition** : Le modèle est trop simple et n'apprend pas assez.  
**Symptôme** : Accuracy Train ET Accuracy Validation faibles.  
**Solutions** : Modèle plus complexe, plus d'epochs, meilleur optimizer.

### Early Stopping
**Définition** : Arrête l'entraînement si la validation loss n'améliore plus pendant N epochs.  
**But** : Éviter l'overfitting.  
**Paramètre** : Patience = nombre d'epochs sans amélioration avant arrêt.  
**Dans Zoidberg** : Patience = 10 epochs utilisée.

### Regularization (L1, L2)
**Définition** : Pénalise les poids trop grands pour éviter l'overfitting.  
**Types** :
- **L1** : Pénalise somme des poids (favorise sparsité).
- **L2** : Pénalise somme des carrés des poids (favorise petits poids).  
**Dans Zoidberg** : Pas utilisé (Dropout suffisant).

---

## 7. Transfer Learning & Fine-Tuning

### Transfer Learning
**Définition** : Réutiliser un modèle pré-entraîné sur un gros dataset (ImageNet) et l'adapter à notre problème.  
**Analogie** : Embaucher un médecin expérimenté plutôt que former un débutant.  
**Avantage** : Apprend plus vite avec moins de données.  
**Dans Zoidberg** : EfficientNetB0 pré-entraîné sur ImageNet (1.4M images).

### Fine-Tuning
**Définition** : Débloquer certaines couches du modèle pré-entraîné pour les réentraîner.  
**Stratégie** :
1. Geler toutes les couches (Feature Extractor)
2. Entraîner uniquement les couches finales
3. Débloquer progressivement des couches et réentraîner (Fine-Tuning)  
**Dans Zoidberg** : On "gèle" les couches de base d'EfficientNet et entraîne uniquement le classifieur final.

### ImageNet
**Définition** : Dataset de 1.4 million d'images dans 1000 catégories (chiens, voitures, etc.).  
**Rôle** : Dataset de référence pour pré-entraîner les modèles de vision.  
**Dans Zoidberg** : EfficientNet pré-entraîné sur ImageNet → connaît déjà les formes, textures, bords.

### Feature Extraction
**Définition** : Utiliser un modèle pré-entraîné comme extracteur de caractéristiques (features) sans réentraîner.  
**Analogie** : Utiliser les lunettes d'un expert pour voir, mais prendre la décision finale soi-même.  
**Dans Zoidberg** : Les couches convolutionnelles d'EfficientNet extraient les features, notre classifieur décide.

---

## 8. Validation & Tests

### Train-Validation-Test Split
**Définition** : Découpage du dataset en 3 ensembles :
- **Train** (70-80%) : Entraînement du modèle
- **Validation** (10-20%) : Ajustement hyperparamètres + Early Stopping
- **Test** (10-20%) : Évaluation finale (jamais vu pendant entraînement)  
**Dans Zoidberg** : Train 70% / Validation 20% / Test 10% (stratifié par classe).

### K-Fold Cross-Validation
**Définition** : Diviser le dataset en K parties, entraîner K fois en utilisant chaque partie comme validation.  
**Avantage** : Estimation plus robuste des performances (variance réduite).  
**Inconvénient** : K fois plus coûteux en calcul.  
**Dans Zoidberg** : **NON utilisé car trop coûteux** (5-Fold = 5× le temps d'entraînement). Dataset assez grand (5,840 images) pour que Train-Val-Test suffise.

### Stratified Split
**Définition** : Découpage qui préserve les proportions de classes dans Train, Val et Test.  
**Exemple** : Si 47% Bactéries dans dataset → 47% Bactéries dans chaque split.  
**Dans Zoidberg** : Utilisé via `class_mode` de Keras.

### Bootstrap Confidence Interval
**Définition** : Méthode de rééchantillonnage pour estimer l'incertitude d'une métrique.  
**Principe** : Tirer N échantillons avec remise, calculer métrique N fois, prendre percentiles.  
**Utilité** : Donner intervalle de confiance (ex: "Accuracy = 71% [68%, 74%]").  
**Dans Zoidberg** : Implémenté dans `advanced_evaluation.py`.

---

## 9. Explicabilité (XAI)

### Grad-CAM (Gradient-weighted Class Activation Mapping)
**Définition** : Technique de visualisation montrant quelles zones de l'image le modèle regarde pour prendre sa décision.  
**Analogie** : Eye-tracking d'un radiologue → où pose-t-il son regard ?  
**Principe** : Calculer les gradients de la classe prédite par rapport à la dernière couche conv → Heatmap.  
**Dans Zoidberg** : **Implémenté et intégré** dans MASTER notebook. Montre que le modèle se concentre sur les opacités pulmonaires (pas les bords de l'image).

### Explainability (Explicabilité)
**Définition** : Capacité à expliquer POURQUOI le modèle a pris telle décision.  
**Importance Médicale** : **CRITIQUE** → Un médecin ne peut pas utiliser une "boîte noire".  
**Techniques** : Grad-CAM, LIME, SHAP, Attention Maps.  
**Dans Zoidberg** : Grad-CAM utilisé pour valider que le modèle ne triche pas (ex: ne regarde pas les métadonnées).

### Black Box (Boîte Noire)
**Définition** : Modèle dont les décisions ne sont pas compréhensibles par un humain.  
**Problème** : Inacceptable en médecine (responsabilité légale, confiance).  
**Solution** : Techniques d'explicabilité (Grad-CAM, etc.).

---

## 10. Termes Médicaux

### Pneumonie
**Définition** : Infection des poumons causée par bactéries, virus ou champignons.  
**Symptômes** : Fièvre, toux, difficultés respiratoires.  
**Gravité** : Peut être mortelle (surtout enfants, personnes âgées).  
**Dans Zoidberg** : Problème médical central à détecter et classifier.

### Opacité Pulmonaire
**Définition** : Zone blanche/grise sur une radiographie thoracique indiquant une anomalie (liquide, inflammation).  
**Types** :
- **Consolidation** : Zone dense, souvent bactérienne.
- **Opacités diffuses** : Motif en verre dépoli, souvent viral.  
**Dans Zoidberg** : Le modèle apprend à détecter ces opacités via Grad-CAM.

### Pneumonie Bactérienne
**Définition** : Causée par bactéries (Streptococcus pneumoniae, etc.).  
**Traitement** : Antibiotiques.  
**Radiographie** : Consolidation dense, localisée.  
**Dans Zoidberg** : 47% du dataset, Recall 98% (excellent).

### Pneumonie Virale
**Définition** : Causée par virus (Influenza, RSV, COVID-19, etc.).  
**Traitement** : Symptomatique, antiviraux parfois.  
**Radiographie** : Opacités diffuses, moins denses.  
**Dans Zoidberg** : 26% du dataset, Recall 45% (difficile, limite médicale connue).

### Faux Positif (FP)
**Définition** : Le modèle prédit "Malade" mais la personne est saine.  
**Conséquence** : Stress patient, examens complémentaires inutiles, coût.  
**Trade-off** : En médecine, on préfère quelques FP que rater un vrai malade (FN).

### Faux Négatif (FN)
**Définition** : Le modèle prédit "Sain" mais la personne est malade.  
**Conséquence** : **DANGEREUX** → Retard de traitement, aggravation.  
**Dans Zoidberg** : Recall 98% = seulement 2% de FN (acceptable cliniquement).

### Triage Médical
**Définition** : Processus de priorisation des patients selon la gravité.  
**Dans Zoidberg** : EfficientNet Binary (AUC 0.9897) = Outil de triage rapide pour détecter les pneumonies.

### Gold Standard
**Définition** : Méthode de référence pour diagnostic (ex: culture bactérienne, PCR).  
**Dans Zoidberg** : Nos labels (vérité terrain) viennent de radiologues experts.

---

## 🔗 Ressources Complémentaires

### Pour aller plus loin

**Deep Learning** :
- Cours : [Deep Learning Specialization (Andrew Ng)](https://www.deeplearning.ai/)
- Livre : "Deep Learning" de Goodfellow, Bengio, Courville

**Computer Vision** :
- Cours : [CS231n Stanford](http://cs231n.stanford.edu/)
- Papier : "ImageNet Classification with Deep CNNs" (AlexNet)

**Transfer Learning** :
- Papier : "A Survey on Transfer Learning" (Pan & Yang)
- Tutoriel : [Transfer Learning Guide (TensorFlow)](https://www.tensorflow.org/tutorials/images/transfer_learning)

**Grad-CAM** :
- Papier original : "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)
- Implémentation : [Keras-CAM](https://github.com/jacobgil/keras-grad-cam)

**IA Médicale** :
- Papier : "Deep Learning in Medical Imaging" (Litjens et al., 2017)
- Dataset : [Kaggle Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 📝 Comment ce glossaire a été utilisé dans Zoidberg

Ce glossaire a été créé pour :

1. **Préparer la présentation orale** → Répondre aux questions du jury
2. **Documenter les choix techniques** → Justifier pourquoi Recall > Accuracy
3. **Former les membres du groupe** → Harmoniser la compréhension
4. **Expliquer aux non-experts** → Rendre accessible l'IA médicale

**Conseil pour la présentation** : Si le jury pose une question sur un terme, utilisez la structure :
1. Définition simple
2. Analogie vulgarisée
3. Application concrète dans Zoidberg

---

*Glossaire créé le 23 juin 2026 - Projet Zoidberg*  
*Mis à jour avec métriques avancées et cross-validation*
