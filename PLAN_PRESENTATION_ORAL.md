# 🎤 PLAN DE PRÉSENTATION ORALE - PROJET ZOIDBERG

**Durée totale** : 15-20 minutes  
**Format** : PowerPoint / Google Slides  
**Objectif** : Démontrer compétences techniques + démarche scientifique

---

# 📊 STRUCTURE GÉNÉRALE (15-20 slides)

## ⏱️ TIMING RECOMMANDÉ

| Section | Slides | Durée | Contenu |
|---------|--------|-------|---------|
| **Introduction** | 1-3 | 2 min | Contexte + Problématique |
| **Dataset & Méthodologie** | 4-6 | 3 min | Données + Approche |
| **Modèles & Résultats** | 7-13 | 8 min | 5 modèles comparés |
| **Innovation Hiérarchique** | 14-16 | 4 min | Pipeline 2-stages |
| **Explicabilité & Limites** | 17-18 | 2 min | Grad-CAM + Transparence |
| **Conclusion** | 19-20 | 1 min | Synthèse + Perspectives |

**Total** : 20 slides × 1 min = **20 minutes**

---

# 🎯 SLIDE PAR SLIDE

## SLIDE 1 : TITRE
```
┌─────────────────────────────────────────────┐
│                                             │
│         PROJET ZOIDBERG                     │
│   Détection de Pneumonie par               │
│        Deep Learning                        │
│                                             │
│   [Image : Radio thoracique + Logo IA]     │
│                                             │
│   Nicolas BRODBECK                          │
│   Juin 2026                                 │
│                                             │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Bonjour, je m'appelle Nicolas Brodbeck et je vais vous présenter mon projet Zoidberg, un système d'aide au diagnostic de pneumonie par Deep Learning."

**Durée** : 15 secondes

---

## SLIDE 2 : CONTEXTE MÉDICAL
```
┌─────────────────────────────────────────────┐
│  LE PROBLÈME MÉDICAL                        │
│                                             │
│  🏥 PNEUMONIE                               │
│     • 200 millions de cas/an                │
│     • Cause majeure mortalité infantile     │
│     • Diagnostic rapide = survie            │
│                                             │
│  🔬 DÉFI DU RADIOLOGUE                      │
│     1. Poumon sain ou malade ?              │
│     2. Origine : Bactérie ou Virus ?        │
│                                             │
│  💊 ENJEU                                   │
│     • Bactérie → Antibiotiques              │
│     • Virus → Antiviraux/Repos              │
│     • Mauvais diagnostic = Danger           │
└─────────────────────────────────────────────┘
```

**À dire** :
> "La pneumonie touche 200 millions de personnes par an. Le radiologue doit répondre à 2 questions : le poumon est-il malade ? Et si oui, est-ce une bactérie ou un virus ? C'est crucial car le traitement est différent."

**Durée** : 45 secondes

---

## SLIDE 3 : PROBLÉMATIQUE
```
┌─────────────────────────────────────────────┐
│  PROBLÉMATIQUE                              │
│                                             │
│  ❓ QUESTION CENTRALE                       │
│     "L'IA peut-elle aider le radiologue     │
│      à détecter et classifier la            │
│      pneumonie ?"                           │
│                                             │
│  🎯 OBJECTIFS                               │
│     ✅ Détecter pneumonie (AUC > 0.95)      │
│     ✅ Différencier Bactérie/Virus (>65%)   │
│     ✅ Assurer explicabilité (Grad-CAM)     │
│                                             │
│  ⚠️  LIMITE CONNUE                          │
│     • Distinction Virus/Bactérie difficile  │
│     • Radiographie seule insuffisante       │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Mon objectif : créer une IA qui assiste le radiologue. Je me suis fixé 3 critères : AUC supérieur à 0.95 pour la détection, accuracy supérieure à 65% pour la classification 3-classes, et explicabilité via Grad-CAM. J'ai conscience de la limite médicale : distinguer virus et bactérie sur radio seule est extrêmement difficile."

**Durée** : 1 minute

---

## SLIDE 4 : DATASET
```
┌─────────────────────────────────────────────┐
│  DATASET                                    │
│                                             │
│  📊 KAGGLE CHEST X-RAY                      │
│     • 5,840 radiographies thoraciques       │
│     • 3 classes : Normal / Bactérie / Virus │
│                                             │
│  📈 DISTRIBUTION                            │
│     [Graphique en barres]                   │
│     Normal   : 1,577 (27%) ████████         │
│     Bactérie : 2,774 (47%) ███████████████  │
│     Virus    : 1,489 (26%) ████████         │
│                                             │
│  ✂️ SPLIT                                   │
│     • Train      : 74%                      │
│     • Validation : 2%                       │
│     • Test       : 24% (1,402 images)       │
│                                             │
│  ⚠️  Déséquilibre classes → Class Weights   │
└─────────────────────────────────────────────┘
```

**À dire** :
> "J'ai utilisé le dataset Kaggle Chest X-Ray : 5,840 radios réparties en 3 classes. Attention au déséquilibre : 47% de bactéries contre 26% de virus. Pour compenser, j'ai appliqué des class weights pendant l'entraînement. Split classique : 74% train, 24% test."

**Durée** : 1 minute

---

## SLIDE 5 : MÉTHODOLOGIE
```
┌─────────────────────────────────────────────┐
│  APPROCHE SCIENTIFIQUE PROGRESSIVE          │
│                                             │
│  Étape 1 : CNN FROM SCRATCH                 │
│             ↓ Baseline + comprendre         │
│  Étape 2 : EFFICIENTNET BINARY              │
│             ↓ Transfer Learning             │
│  Étape 3 : EFFICIENTNET MULTI-CLASSES       │
│             ↓ Approche directe 3-classes    │
│  Étape 4 : PIPELINE HIÉRARCHIQUE            │
│             ↓ INNOVATION (diviser)          │
│  Étape 5 : EXPERT SOUS-TYPE                 │
│             ↓ Spécialisation                │
│                                             │
│  🧠 PHILOSOPHIE                             │
│     • Rigueur scientifique                  │
│     • Comparaison systématique              │
│     • Métriques médicales (Recall)          │
└─────────────────────────────────────────────┘
```

**À dire** :
> "J'ai adopté une démarche progressive en 5 étapes. Partir d'une baseline CNN from scratch, explorer le Transfer Learning, tester l'approche directe 3-classes, puis innover avec un pipeline hiérarchique. Toujours avec des métriques adaptées au médical : le Recall est prioritaire sur l'Accuracy."

**Durée** : 1 minute

---

## SLIDE 6 : MÉTRIQUES MÉDICALES
```
┌─────────────────────────────────────────────┐
│  POURQUOI RECALL > ACCURACY ?               │
│                                             │
│  ⚖️ LES 2 TYPES D'ERREURS                   │
│                                             │
│  ❌ FAUX POSITIF                            │
│     Dire "malade" alors que sain            │
│     → Tests inutiles                        │
│     → Désagréable mais PAS GRAVE            │
│                                             │
│  🔴 FAUX NÉGATIF                            │
│     Dire "sain" alors que malade            │
│     → Patient malade rentre chez lui        │
│     → DANGER !!!                            │
│                                             │
│  ✅ CHOIX MÉDICAL                           │
│     RECALL 98% = Ne rate que 2% malades     │
│     → Acceptable cliniquement               │
│                                             │
│     Recall = TP / (TP + FN)                 │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Pourquoi je privilégie le Recall ? En médical, un faux positif c'est désagréable mais un faux négatif c'est dangereux. Mon modèle atteint 98% de Recall : il ne rate que 2% des malades. C'est le bon compromis pour un outil d'aide au diagnostic."

**Durée** : 1 minute

---

## SLIDE 7 : MODÈLE 1 - CNN FROM SCRATCH
```
┌─────────────────────────────────────────────┐
│  MODÈLE 1 : CNN FROM SCRATCH                │
│                                             │
│  🏗️ ARCHITECTURE                            │
│     Conv → ReLU → MaxPool (×4)              │
│     → Flatten → Dense → Softmax             │
│     Paramètres : 5M                         │
│                                             │
│  📊 RÉSULTATS                               │
│     • Accuracy : 67.15%                     │
│     • AUC      : 0.9924 ⭐                  │
│     • Recall Virus : 45.3%                  │
│                                             │
│  ✅ FORCES                                  │
│     • AUC excellent (capacité discriminante)│
│     • Baseline solide                       │
│                                             │
│  ❌ FAIBLESSES                              │
│     • Accuracy limitée (67%)                │
│     • Confusion Virus/Bactérie              │
│                                             │
│  💡 LEÇON                                   │
│     Transfer Learning nécessaire            │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Mon premier modèle, un CNN from scratch. Architecture classique avec convolutions et pooling. Résultat : 67% d'accuracy mais un AUC de 0.99, ce qui est excellent. Le modèle sait discriminer les classes mais confond virus et bactéries. Leçon : il faut du Transfer Learning."

**Durée** : 1 minute

---

## SLIDE 8 : MODÈLE 2 - EFFICIENTNET BINARY
```
┌─────────────────────────────────────────────┐
│  MODÈLE 2 : EFFICIENTNET BINARY             │
│                                             │
│  🚀 TRANSFER LEARNING                       │
│     • EfficientNetB0 pré-entraîné (ImageNet)│
│     • Fine-tuning : 20 dernières couches    │
│     • 5.3M paramètres (1.3M entraînables)   │
│                                             │
│  📊 RÉSULTATS                               │
│     • AUC      : 0.9897 ⭐⭐⭐              │
│     • Recall   : 98.0% 🔥                   │
│     • Precision: 88.0%                      │
│                                             │
│  [Graphique ROC Curve]                      │
│                                             │
│  ✅ PERFORMANCE CLINIQUE                    │
│     Ne rate que 2% des malades              │
│     Prêt pour prototype médical             │
│                                             │
│  ❌ LIMITE                                  │
│     Classification binaire uniquement       │
│     (Normal vs Pneumonie)                   │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Modèle 2 : EfficientNet Binary avec Transfer Learning. J'ai réutilisé les poids ImageNet et fine-tuné les 20 dernières couches. Résultat : AUC 0.9897 et Recall 98%. Le modèle ne rate presque aucun malade. C'est cliniquement excellent, mais il ne différencie pas bactérie et virus."

**Durée** : 1 minute

---

## SLIDE 9 : MODÈLE 3 - EFFICIENTNET MULTI-CLASSES
```
┌─────────────────────────────────────────────┐
│  MODÈLE 3 : EFFICIENTNET MULTI-CLASSES      │
│                                             │
│  🎯 OBJECTIF                                │
│     Classifier directement en 3 classes     │
│     (Normal / Bactérie / Virus)             │
│                                             │
│  📊 RÉSULTATS                               │
│     • Accuracy : 56.28% ⚠️                  │
│     • F1-Macro : 54.97%                     │
│     • Recall Virus : 41.0%                  │
│                                             │
│  [Matrice de Confusion]                     │
│  → Confusion généralisée                    │
│                                             │
│  ❌ ÉCHEC RELATIF                           │
│     Pire que CNN from scratch (67%)         │
│                                             │
│  💡 ANALYSE                                 │
│     • Modèle apprend 2 tâches simultanément │
│     • Gradients se mélangent                │
│     • Besoin nouvelle approche              │
│                                             │
│  ➡️ IDÉE : Diviser le problème              │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Modèle 3 : approche directe 3-classes. Surprise : seulement 56% d'accuracy, pire que le CNN from scratch ! Pourquoi ? Le modèle doit apprendre 2 tâches en même temps : détecter l'anomalie ET la qualifier. Ça ne marche pas. Il faut changer de stratégie : diviser le problème."

**Durée** : 1 minute 15 secondes

---

## SLIDE 10 : INNOVATION - PIPELINE HIÉRARCHIQUE
```
┌─────────────────────────────────────────────┐
│  💡 INNOVATION : PIPELINE HIÉRARCHIQUE      │
│                                             │
│  🧠 RAISONNEMENT MÉDICAL                    │
│     Radiologue ne fait PAS :                │
│     "Normal OU Bactérie OU Virus"           │
│                                             │
│     Il fait :                               │
│     1. "Y a-t-il une anomalie ?" (Binaire)  │
│     2. "Si oui, quelle nature ?" (Binaire)  │
│                                             │
│  ➡️ IMITER L'EXPERT                         │
│                                             │
│  🔧 ARCHITECTURE 2-STAGES                   │
│                                             │
│     STAGE 1 : EfficientNet Binary           │
│         Normal vs Pneumonie                 │
│         AUC 0.9897, Recall 98%              │
│              ↓                              │
│     STAGE 2 : Expert Sous-type              │
│         Bactérie vs Virus                   │
│         (uniquement sur pneumonies)         │
│         Accuracy 76%                        │
│                                             │
│  ✅ DIVISER POUR MIEUX RÉGNER               │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Mon innovation : le pipeline hiérarchique. J'ai imité le raisonnement médical réel. Un radiologue ne choisit pas entre 3 options d'un coup, il procède par étapes : d'abord détecter l'anomalie, ensuite la qualifier. J'ai créé 2 modèles spécialisés qui travaillent en cascade."

**Durée** : 1 minute 30 secondes

---

## SLIDE 11 : FLOW DIAGRAM PIPELINE
```
┌─────────────────────────────────────────────┐
│  FLUX DU PIPELINE HIÉRARCHIQUE              │
│                                             │
│  [Image : hierarchical_flow_diagram.png]    │
│                                             │
│        1402 Radios Test                     │
│              ↓                              │
│     ┌─────────────────┐                     │
│     │ STAGE 1 BINAIRE │                     │
│     │  AUC: 0.9897    │                     │
│     └────────┬─────────┘                    │
│              │                              │
│       ┌──────┴──────┐                       │
│       ▼             ▼                       │
│    NORMAL      PNEUMONIE                    │
│    52%          48%                         │
│                  ↓                          │
│          ┌──────────────┐                   │
│          │ STAGE 2      │                   │
│          │ EXPERT       │                   │
│          └──────┬───────┘                   │
│                 │                           │
│          ┌──────┴──────┐                    │
│          ▼             ▼                    │
│      BACTÉRIE       VIRUS                   │
│        95%           5%                     │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Voici le flux complet. Les 1,402 images test passent d'abord par le Stage 1 qui détecte 48% de pneumonies. Ces cas passent ensuite au Stage 2 qui différencie bactérie et virus. Chaque modèle est expert sur SA tâche."

**Durée** : 45 secondes

---

## SLIDE 12 : RÉSULTATS PIPELINE
```
┌─────────────────────────────────────────────┐
│  RÉSULTATS PIPELINE HIÉRARCHIQUE            │
│                                             │
│  📊 MÉTRIQUES GLOBALES (3 classes)          │
│     • Accuracy : 71.0% 🔥                   │
│     • F1-Macro : 65.0%                      │
│                                             │
│  📈 PAR CLASSE                              │
│     Normal   : Recall 52%, Precision 70%    │
│     Bactérie : Recall 95%, Precision 72%    │
│     Virus    : Recall 45%, Precision 79%    │
│                                             │
│  ✅ AMÉLIORATION                            │
│     +15 POINTS vs Multi-classes direct      │
│     56% → 71%                               │
│                                             │
│  🏆 MEILLEUR MODÈLE GLOBAL                  │
│     • Recall Bactérie 95% (ne rate rien)    │
│     • Approche innovante validée            │
│                                             │
│  ⚠️  LIMITE                                 │
│     Recall Virus toujours à 45%             │
│     (limite médicale connue)                │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Résultats du pipeline : 71% d'accuracy, soit +15 points par rapport à l'approche directe. C'est mon meilleur modèle global. Le Recall Bactérie atteint 95%, ce qui est excellent pour les cas sévères. Le Recall Virus reste à 45%, mais c'est une limite médicale documentée : la radio seule ne suffit pas."

**Durée** : 1 minute 15 secondes

---

## SLIDE 13 : TABLEAU COMPARATIF
```
┌─────────────────────────────────────────────┐
│  COMPARAISON TOUS LES MODÈLES               │
│                                             │
│  [Tableau]                                  │
│  ┌──────────────┬──────┬────┬─────────┐    │
│  │ Modèle       │ Acc  │AUC │ Recall  │    │
│  ├──────────────┼──────┼────┼─────────┤    │
│  │CNN Scratch   │67.2% │0.99│  57%    │    │
│  │Eff. Binary   │  -   │0.99│  98%    │    │
│  │Eff. Multi    │56.3% │ -  │  60%    │    │
│  │PIPELINE      │71.0% │ -  │  95%    │    │
│  │Expert Subtype│76.0% │ -  │  98%    │    │
│  └──────────────┴──────┴────┴─────────┘    │
│                                             │
│  [Graphique barres Accuracy]                │
│                                             │
│  🏆 PODIUM                                  │
│     🥇 Accuracy : Pipeline (71%)            │
│     🥇 AUC      : CNN (0.9924)              │
│     🥇 Recall   : Binaire (98%)             │
│                                             │
│  💡 INSIGHT                                 │
│     Diviser problème > Approche directe     │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Comparaison des 5 modèles. Le pipeline hiérarchique est le grand gagnant avec 71% d'accuracy. L'EfficientNet Binary a le meilleur Recall à 98%. Le CNN from scratch a le meilleur AUC. Chaque modèle a sa force, mais le pipeline est le plus équilibré pour une utilisation clinique complète."

**Durée** : 1 minute

---

## SLIDE 14 : EXPLICABILITÉ - GRAD-CAM
```
┌─────────────────────────────────────────────┐
│  EXPLICABILITÉ : GRAD-CAM                   │
│                                             │
│  ❓ PROBLÈME                                │
│     CNN = Boîte noire                       │
│     Médecin ne peut pas faire confiance     │
│                                             │
│  💡 SOLUTION : GRAD-CAM                     │
│     Gradient-weighted Class Activation Map  │
│                                             │
│  [Image : grad_cam_bacteria_example.png]    │
│     ┌──────┬──────┬───────────┐            │
│     │Radio │Heatmap│Superposé │            │
│     └──────┴──────┴───────────┘            │
│                                             │
│  🔍 INTERPRÉTATION                          │
│     Zones rouges = Zones regardées          │
│                                             │
│  ✅ BONNE PRÉDICTION                        │
│     Activation sur les poumons              │
│                                             │
│  ❌ SHORTCUT LEARNING                       │
│     Activation sur lettres L/R              │
│     → Détecté et corrigé (cropping)         │
│                                             │
│  🏥 IMPACT CLINIQUE                         │
│     Radiologue peut valider visuellement    │
└─────────────────────────────────────────────┘
```

**À dire** :
> "L'explicabilité est cruciale en médical. J'ai intégré Grad-CAM qui génère des cartes de chaleur montrant où le modèle regarde. Zones rouges = zones importantes pour la décision. Cela permet au radiologue de valider que l'IA analyse bien les poumons et pas des artefacts. J'ai d'ailleurs détecté du shortcut learning sur les lettres L/R et corrigé avec du cropping."

**Durée** : 1 minute 30 secondes

---

## SLIDE 15 : LIMITES & TRANSPARENCE
```
┌─────────────────────────────────────────────┐
│  LIMITES & TRANSPARENCE SCIENTIFIQUE        │
│                                             │
│  ⚠️  3 LIMITES ASSUMÉES                     │
│                                             │
│  1️⃣ DISTINCTION VIRUS/BACTÉRIE (45%)        │
│     • Limite médicale connue                │
│     • Opacités se ressemblent visuellement  │
│     • Radio seule insuffisante              │
│     → Solution : Intégrer données cliniques │
│                                             │
│  2️⃣ DATASET DÉSÉQUILIBRÉ                    │
│     • 47% Bactérie vs 26% Virus             │
│     • Biais vers classe majoritaire         │
│     → Solution : Class weights appliqués    │
│                                             │
│  3️⃣ IA = OUTIL D'AIDE, PAS REMPLACEMENT     │
│     • Décision finale au médecin            │
│     • Validation Grad-CAM nécessaire        │
│     • Confiance + Expertise humaine         │
│                                             │
│  ✅ PHILOSOPHIE                             │
│     Transparence > Performance gonflée      │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Je suis transparent sur les limites. Première limite : la distinction virus/bactérie ne dépasse pas 45% de Recall. C'est une limite médicale documentée. Deuxième limite : dataset déséquilibré que j'ai compensé avec des class weights. Troisième limite : l'IA est un outil d'aide, pas un remplacement du médecin. La décision finale lui revient toujours."

**Durée** : 1 minute 15 secondes

---

## SLIDE 16 : CAS D'USAGE CLINIQUE
```
┌─────────────────────────────────────────────┐
│  RECOMMANDATIONS PAR CAS D'USAGE            │
│                                             │
│  🏥 TRIAGE RAPIDE (Urgences)                │
│     → EfficientNet Binary                   │
│     • Recall 98% (ne rate rien)             │
│     • Rapide (1 modèle)                     │
│     • Workflow : Normal → Rentre            │
│                  Pneumonie → Spécialiste    │
│                                             │
│  🏥 DIAGNOSTIC COMPLET (Hospitalisation)    │
│     → Pipeline Hiérarchique                 │
│     • 71% accuracy (3 classes)              │
│     • Recall Bactérie 95%                   │
│     • Grad-CAM pour validation              │
│                                             │
│  🔬 RECHERCHE SCIENTIFIQUE                  │
│     → Expert Sous-type                      │
│     • 76% accuracy max                      │
│     • Precision Virus 94%                   │
│     • Études épidémiologiques               │
│                                             │
│  💡 CHOIX SELON CONTEXTE                    │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Mes recommandations par cas d'usage. Pour un triage rapide aux urgences : EfficientNet Binary avec son Recall de 98%. Pour un diagnostic complet à l'hôpital : le Pipeline Hiérarchique. Pour la recherche : l'Expert Sous-type. Le bon modèle dépend du contexte clinique."

**Durée** : 1 minute

---

## SLIDE 17 : PERSPECTIVES
```
┌─────────────────────────────────────────────┐
│  PERSPECTIVES D'AMÉLIORATION                │
│                                             │
│  📅 COURT TERME (3-6 mois)                  │
│     • Focal Loss (équilibrer classes)       │
│     • Ensemble Learning (voter)             │
│     • Test-Time Augmentation                │
│     → Impact : Recall Virus +10-15%         │
│                                             │
│  📅 MOYEN TERME (6-12 mois)                 │
│     • Vision Transformers (SOTA)            │
│     • Multimodalité (symptômes + radio)     │
│     • Segmentation pulmonaire (U-Net)       │
│     → Impact : Accuracy +10-15%             │
│                                             │
│  📅 LONG TERME (1-2 ans)                    │
│     • Migration PyTorch                     │
│     • Déploiement API REST                  │
│     • Étude clinique (validation réelle)    │
│     • Publication scientifique              │
│     → Impact : Certification médicale       │
│                                             │
│  🎯 OBJECTIF FINAL                          │
│     Outil d'aide au diagnostic certifié     │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Perspectives à 3 niveaux temporels. Court terme : Focal Loss et Ensemble Learning pour gagner 10-15% de Recall Virus. Moyen terme : Vision Transformers et multimodalité pour intégrer symptômes et analyses. Long terme : étude clinique réelle et certification médicale. L'objectif final : un outil validé et déployable."

**Durée** : 1 minute

---

## SLIDE 18 : COMPÉTENCES DÉMONTRÉES
```
┌─────────────────────────────────────────────┐
│  COMPÉTENCES TECHNIQUES VALIDÉES            │
│                                             │
│  🤖 DEEP LEARNING                           │
│     ✅ CNN from scratch (architecture)      │
│     ✅ Transfer Learning (fine-tuning)      │
│     ✅ Optimisation (hyperparamètres)       │
│     ✅ Explicabilité (Grad-CAM)             │
│                                             │
│  📊 DATA SCIENCE                            │
│     ✅ Preprocessing (normalisation)        │
│     ✅ Augmentation (rotation, zoom)        │
│     ✅ Class weights (déséquilibre)         │
│     ✅ Métriques médicales (AUC, Recall)    │
│                                             │
│  🏗️ SOFTWARE ENGINEERING                    │
│     ✅ Git (versioning propre)              │
│     ✅ Python (TensorFlow, NumPy, Pandas)   │
│     ✅ Pipeline automatisé                  │
│     ✅ Documentation exhaustive             │
│                                             │
│  🧠 SOFT SKILLS                             │
│     ✅ Pensée critique (analyser limites)   │
│     ✅ Communication (18 docs markdown)     │
│     ✅ Rigueur scientifique (5 modèles)     │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Ce projet m'a permis de démontrer un large éventail de compétences. En Deep Learning : CNN, Transfer Learning, optimisation. En Data Science : preprocessing, augmentation, métriques adaptées. En Software Engineering : Git, Python, pipelines. Et en soft skills : pensée critique sur les limites et communication via 18 documents markdown."

**Durée** : 1 minute

---

## SLIDE 19 : CONCLUSION
```
┌─────────────────────────────────────────────┐
│  CONCLUSION                                 │
│                                             │
│  ✅ OBJECTIFS ATTEINTS                      │
│     🎯 AUC > 0.95      → 0.9897 ✅          │
│     🎯 Accuracy > 65%  → 71.0% ✅           │
│     🎯 Explicabilité   → Grad-CAM ✅        │
│                                             │
│  🏆 CONTRIBUTIONS                           │
│     • 5 modèles comparés rigoureusement     │
│     • Innovation : Pipeline Hiérarchique    │
│     • Gain : +15% vs approche directe       │
│     • Transparence sur limites              │
│                                             │
│  💡 IMPACT POTENTIEL                        │
│     • Prototype cliniquement viable         │
│     • Aide au diagnostic (pas remplacement) │
│     • Recall 98% = Sécurité patient         │
│                                             │
│  📈 PROCHAINES ÉTAPES                       │
│     • Validation clinique                   │
│     • Publication scientifique              │
│     • Déploiement réel                      │
│                                             │
│  "L'IA au service du médecin,               │
│   pas à sa place."                          │
└─────────────────────────────────────────────┘
```

**À dire** :
> "En conclusion, j'ai atteint tous mes objectifs : AUC 0.99, Accuracy 71%, explicabilité via Grad-CAM. Ma contribution principale : le pipeline hiérarchique qui améliore de 15% les performances. Le projet est transparent sur ses limites et prêt pour une validation clinique. Mon crédo : l'IA au service du médecin, pas à sa place."

**Durée** : 1 minute 30 secondes

---

## SLIDE 20 : QUESTIONS
```
┌─────────────────────────────────────────────┐
│                                             │
│                                             │
│            MERCI DE VOTRE ATTENTION         │
│                                             │
│                 Questions ?                 │
│                                             │
│                                             │
│         [Image : Logo projet]               │
│                                             │
│                                             │
│    📧 nbrodbeck@email.com                   │
│    🐙 github.com/nbrodbeck/zoidberg         │
│                                             │
│                                             │
└─────────────────────────────────────────────┘
```

**À dire** :
> "Merci de votre attention. Je suis prêt à répondre à vos questions."

**Durée** : 10 secondes

---

# 🎯 QUESTIONS FRÉQUENTES (PRÉPARATION)

## Question 1 : "Pourquoi EfficientNet plutôt que ResNet ?"
**Réponse** :
> "EfficientNet est 10× plus léger que ResNet (5M vs 60M paramètres) pour des performances équivalentes. Avec un dataset médical limité à 5,840 images, un modèle compact réduit le risque d'overfitting. De plus, l'entraînement est 2× plus rapide, ce qui m'a permis d'itérer plus rapidement."

---

## Question 2 : "Comment avez-vous géré le déséquilibre des classes ?"
**Réponse** :
> "Trois techniques : Class weights calculés avec sklearn (pénalisent davantage les erreurs sur classes minoritaires), Data augmentation plus agressive sur la classe Virus, et choix de métriques adaptées (Recall par classe plutôt que Accuracy globale)."

---

## Question 3 : "Pourquoi le Recall Virus est si bas (45%) ?"
**Réponse** :
> "C'est une limite médicale documentée. Les pneumonies virales et bactériennes ont des opacités qui se chevauchent visuellement sur radiographie. Un radiologue expert a lui-même une accuracy de 70-75% sur ce dataset. Pour aller au-delà, il faut intégrer des données cliniques : symptômes, analyses sanguines, contexte épidémiologique."

---

## Question 4 : "Qu'est-ce que Grad-CAM techniquement ?"
**Réponse** :
> "Grad-CAM calcule le gradient de la prédiction par rapport aux feature maps de la dernière couche de convolution. Cela donne l'importance de chaque zone de l'image. On pondère les feature maps par ces gradients, on obtient une heatmap qu'on superpose à l'image originale. Zones rouges = zones importantes pour la décision."

---

## Question 5 : "Avez-vous testé d'autres architectures ?"
**Réponse** :
> "J'ai testé 5 approches : CNN from scratch, EfficientNet, et le pipeline hiérarchique. J'ai considéré ResNet et VGG mais ils sont trop lourds pour ce dataset. Dans les perspectives, je mentionne Vision Transformers (ViT) et ConvNeXt qui sont l'état de l'art actuel, mais ils nécessitent plus de données ou un meilleur Transfer Learning."

---

## Question 6 : "Comment déploieriez-vous ce modèle en production ?"
**Réponse** :
> "Architecture en 3 couches : API REST FastAPI qui expose un endpoint /predict, conteneurisation Docker pour portabilité, orchestration Kubernetes pour scalabilité. Le workflow : le radiologue upload la radio via interface web, l'API prédit + génère Grad-CAM, retourne résultat + heatmap. Le radiologue valide ou infirme. Tout est loggé pour amélioration continue."

---

## Question 7 : "Quelles sont les contraintes éthiques ?"
**Réponse** :
> "Trois contraintes majeures : Responsabilité en cas d'erreur (qui est responsable ?), Biais du dataset (principalement occidental, généralisation ?), Explicabilité pour la confiance médicale (Grad-CAM suffisant ?). De plus, certification médicale obligatoire (FDA, marquage CE). Mon approche : transparence totale sur limites, IA comme aide pas remplacement, validation humaine toujours finale."

---

# ✅ CHECKLIST AVANT PRÉSENTATION

## 24h Avant
- [ ] Répéter présentation 3× (chronomètre)
- [ ] Vérifier toutes les images s'affichent
- [ ] Préparer fichier PDF backup (si PowerPoint plante)
- [ ] Tester sur l'ordinateur de présentation
- [ ] Imprimer notes (au cas où)

## 1h Avant
- [ ] Arriver en avance (installer présentation)
- [ ] Tester projecteur / écran
- [ ] Vérifier résolution affichage
- [ ] Avoir clé USB backup
- [ ] Respirer profondément

## Pendant
- [ ] Contact visuel avec jury
- [ ] Parler clairement et posément
- [ ] Pointer les éléments importants slides
- [ ] Gérer le timing (20 min max)
- [ ] Sourire et montrer passion

## Questions
- [ ] Écouter la question complète
- [ ] Reformuler si besoin
- [ ] Répondre structuré (contexte → réponse → conclusion)
- [ ] Admettre si on ne sait pas ("Bonne question, je n'ai pas exploré cet aspect")
- [ ] Ne pas se justifier excessivement

---

# 🎨 CONSEILS DESIGN SLIDES

## Palette Couleurs
```
Primaire  : #2E86AB (Bleu professionnel)
Secondaire: #A23B72 (Magenta pour highlights)
Accent    : #2A9D8F (Vert pour succès)
Attention : #E76F51 (Orange pour warnings)
Texte     : #333333 (Gris foncé)
Fond      : #FFFFFF (Blanc)
```

## Typographie
- **Titres** : Arial Bold, 32pt
- **Sous-titres** : Arial, 24pt
- **Corps** : Arial, 18pt
- **Code/Chiffres** : Courier New, 16pt

## Règles
- ✅ 1 idée par slide
- ✅ Max 6 bullets par slide
- ✅ Graphiques > Texte
- ✅ Images haute résolution (300 DPI)
- ❌ Pas d'animations fancy
- ❌ Pas de murs de texte

---

**🎤 TU ES PRÊT À CARTONNER TON ORAL MON BRO !** 🔥

*Plan de présentation créé le 22 juin 2026*
