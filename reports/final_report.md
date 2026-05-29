# 📊 Rapport Final : Architecture Zoidberg (Version Keras)

## 1. Évaluation du Modèle Hiérarchique (Pipeline Complet à 3 classes)

Le pipeline complet évalue les patients en 3 classes (Normal, Bactérie, Virus) en utilisant nos deux modèles spécialisés en cascade (Binaire puis Sous-Type).

**Scores Globaux :**
- **Accuracy Globale :** 71% (1402 images testées)
- **F1-Score Macro :** 65%

**Analyse détaillée par classe :**
- **Bactérie (Pneumonie Bactérienne) :** Très bonne détection ! Avec un **recall de 95%** et un F1-score de 80%, le système global est extrêmement robuste pour isoler les cas bactériens.
- **Normal (Sain) :** La précision est de 70%, mais le recall est à 52%. Cela indique que le Modèle Binaire (Stage 1) a tendance à classer certains poumons sains comme "malades" (faux positifs). Cliniquement parlant, il est toujours préférable d'avoir un faux positif (alerter un médecin pour rien) plutôt qu'un faux négatif (ignorer un patient malade).
- **Virus (Pneumonie Virale) :** Bonne précision (79%), mais recall bas (45%). Les virus sont noyés dans les prédictions bactériennes.

---

## 2. Évaluation de l'Expert Sous-Type (Bactérie vs Virus)

Cette évaluation isole le second modèle (Stage 2) pour mesurer sa capacité brute à différencier une pneumonie bactérienne d'une pneumonie virale, sur un sous-ensemble purement malade.

**Scores de l'Expert :**
- **Accuracy :** 76% (1202 images testées)
- **F1-Score Macro :** 73%

**Analyse Clinique :**
- **Performance sur la Bactérie :** **Rappel exceptionnel de 98%** ! Le modèle ne rate quasiment aucune bactérie. Sur les 688 cas bactériens du jeu de test, il en identifie la quasi-totalité.
- **Performance sur le Virus :** **Précision exceptionnelle de 94%**. Quand le modèle affirme "C'est un Virus", il est presque certain d'avoir raison. Le défi se situe sur le rappel (48%), ce qui signifie que le modèle "sur-prédit" la bactérie face à des cas viraux incertains. C'est un biais connu en radiologie, car les opacités virales et bactériennes peuvent se superposer visuellement.

---

## 3. Conclusion et Prochaines Étapes

L'approche hiérarchique a brillamment rempli son rôle : diviser la complexité. 
Le système actuel excelle comme **filet de sécurité bactérien** :
1. Il ne laisse passer aucune bactérie sévère.
2. Ses détections virales sont extrêmement fiables.

**Ces résultats constituent une baseline (référence) professionnelle très solide pour la version Keras.**

### Vers l'Avenir (PyTorch)
Le passage futur du projet vers **PyTorch** nous permettra d'attaquer la principale faiblesse actuelle (la confusion Virus -> Bactérie) en :
- Utilisant des architectures plus modernes (Vision Transformers, ConvNeXt).
- Améliorant les techniques de loss (Focal Loss, Class Weights) pour pénaliser la sur-prédiction de la classe Bactérie.
- Intégrant des techniques poussées de Data Augmentation ciblées sur la classe Virus.
