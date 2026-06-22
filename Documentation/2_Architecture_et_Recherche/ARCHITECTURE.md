# 🏛️ Architecture Globale et Conceptuelle - Projet Zoidberg

Ce document ne liste pas simplement les fichiers, il explique **pourquoi** Zoidberg est construit de cette façon. Il détaille la réflexion scientifique, les problématiques médicales rencontrées, et les solutions algorithmiques (l'architecture) mises en place pour y répondre.

---

## 🎯 1. La Problématique Initiale
**La question métier :** Comment assister un médecin ou un radiologue pour diagnostiquer rapidement et précisément une pneumonie à partir d'une simple radiographie du thorax, et en déterminer l'origine (Virale ou Bactérienne) pour adapter le traitement ?

**Le défi technique :** Les radiographies médicales sont des images complexes. Les zones infectées (opacités) sont parfois très subtiles. De plus, les datasets médicaux sont souvent "déséquilibrés" (il y a souvent plus d'exemples d'une maladie que d'une autre).

---

## 💾 2. Le Pipeline de Données (Data Ingestion & Preprocessing)
*Localisation : `src/data/data_loader_v3.py`*

**La question posée :** Comment nourrir une Intelligence Artificielle avec des images de tailles différentes, parfois trop peu nombreuses, et s'assurer qu'elle apprenne de manière générale sans "apprendre par cœur" ?

**L'Architecture mise en place :**
- **Normalisation** : Toutes les images sont converties sur une échelle de pixels de 0 à 1. *Pourquoi ?* Les réseaux de neurones convergent beaucoup plus vite et de manière plus stable avec de petites valeurs.
- **Data Augmentation** : Le module génère "à la volée" de nouvelles images pendant l'entraînement en effectuant de légères rotations, des zooms, ou des retournements (flips) horizontaux. *Pourquoi ?* Pour forcer l'IA à reconnaître la maladie peu importe l'angle de la radio, évitant ainsi le "sur-apprentissage" (overfitting).
- **Center Cropping (Rognage central)** : Les images subissent un léger zoom au centre. *Pourquoi ?* Pour éliminer les bords de la radio (qui contiennent souvent des lettres métalliques 'L' ou 'R', ou du texte médical) qui pourraient biaiser l'IA.

---

## 🧠 3. Les Moteurs de Décision (L'Évolution des Modèles)
*Localisation : `src/models/`*

Au lieu de faire un seul "gros modèle magique", l'architecture a évolué par itérations pour résoudre des problèmes de plus en plus complexes.

### A. La Baseline (Le point de repère)
- **Pourquoi ?** Avant de sortir l'artillerie lourde (Deep Learning), il faut prouver que le problème est complexe.
- **Comment ?** `baseline_model.py` écrase les images et utilise des mathématiques classiques (PCA + Régression Logistique). Ses performances modestes justifient l'utilisation des Réseaux de Neurones.

### B. Le CNN Binaire (La sécurité médicale avant tout)
- **La question :** Le patient est-il malade ou en bonne santé ?
- **Comment ?** `cnn_model.py`. C'est un réseau de neurones convolutif "Fait Maison" spécialisé dans la distinction "Sain / Pneumonie".
- **Le rôle métier :** Il a été optimisé pour avoir un **Recall extrêmement élevé (98%)**. Dans le milieu médical, on préfère une IA qui fait parfois une fausse alerte (faux positif) plutôt qu'une IA qui rate un patient malade et le renvoie chez lui (faux négatif).

### C. Le Pipeline Hiérarchique (La solution au problème Multi-classes)
- **Le problème rencontré :** Demander à une seule IA de trier en un coup "Normal, Bactérie, ou Virus" donnait de mauvais résultats (~55%). L'IA s'emmêlait les pinceaux.
- **L'Architecture Hiérarchique (`hierarchical_pipeline.py`) :** Plutôt qu'un modèle généraliste, Zoidberg utilise deux "spécialistes" en série :
  1. **L'Expert Binaire** : Il regarde la radio. Si c'est "Normal", on s'arrête là. S'il détecte une anomalie, il passe le relais.
  2. **L'Expert Sous-type (Transfer Learning)** : Il prend la radio malade et utilise l'architecture très profonde **EfficientNet** (pré-entraînée sur des millions d'images) pour extraire des détails infimes et trancher : "C'est Viral" ou "C'est Bactérien".
- **Pourquoi ?** Diviser pour mieux régner. Cela imite le raisonnement humain d'un médecin (On repère l'anomalie d'abord, on la qualifie ensuite).

---

## 👁️ 4. Explicabilité et Fiabilité (eXplainable AI - XAI)
*Localisation : `src/visualization/grad_cam.py` & `src/models/evaluation.py`*

**La question posée :** "C'est bien beau que l'IA dise que c'est une pneumonie, mais comment le médecin peut-il lui faire confiance ? Et si l'IA trichait ?"

**L'Architecture mise en place :**
- **Lutte contre le "Shortcut Learning" (Triche de l'IA)** : Parfois, une IA remarque que toutes les radios des patients malades viennent d'une certaine machine qui laisse une marque sur l'image. Elle se met à détecter la marque au lieu des poumons !
- **La solution Grad-CAM (`grad_cam.py`)** : Ce module génère une "Heatmap" (Carte de chaleur) qui se superpose à la radio. Les zones rouges montrent exactement quels pixels ont poussé l'IA à dire "C'est une pneumonie". 
- **Le rôle métier :** Le médecin regarde la radio, voit la zone rouge de l'IA. Si la zone rouge est bien sur l'opacité du poumon, le médecin valide. Si la zone rouge est sur une clavicule, le médecin sait que l'IA se trompe. C'est la garantie de sécurité de Zoidberg.

---

## 📊 5. Conclusion de l'Architecture
L'application Zoidberg n'est pas qu'un simple script d'entraînement. C'est un **Pipeline MLOps** complet :
1. **Data Loader** sécurisé (anti-biais par crop).
2. **Modèles** découpés en sous-tâches (Hierarchical Pipeline).
3. **Évaluation** exigeante (Matrices de confusion).
4. **Interprétabilité** intégrée (Grad-CAM pour la preuve visuelle).
