# 🧠 Mémoire et Suivi d'Évolution - Zoidberg

Bienvenue dans le "Project Memory". Ce dossier est conçu pour garder une trace chronologique des ajouts, modifications, pivots, et nouvelles fonctionnalités (features) implémentés sur le projet Zoidberg.

## 📝 Comment procéder ?
À chaque ajustement majeur, nouvelle implémentation, ou refonte, une nouvelle section ci-dessous, ou un nouveau fichier détaillé dans ce dossier (`project_memory/`) doit être ajouté avec la date correspondante.

---

## 🕒 Ligne du Temps (Changelog Global)

### [20 Avril 2026] - Initialisation de la mémoire et Scan global
- **Création du répertoire de mémoire** : Ajout du dossier `project_memory` pour assurer un suivi persistant des changements.
- **Récapitulatif de l'état actuel** : Génération du rapport de statuts des modèles (Baseline PCA, CNN Binaire "from scratch", et Transfer Learning avec EfficientNetB0 (Binaire & Multiclass)). 
- **Observations relevées** : Actuellement, la classification binaire (Sain/Pneumonie) est performante (Recall 98%). La classification Multiclass (Sain/Virale/Bactérienne) rencontre encore des plafonds de performance (Macro F1: ~55%).
- **Lien vers le récap complet** : [Consulter l'audit : 00_recap_etat_actuel.md](./00_recap_etat_actuel.md)

### [25 Avril 2026] - Interprétabilité et Lutte contre le "Shortcut Learning"
- **Implémentation de Grad-CAM** : Création de `src/visualization/grad_cam.py` pour visualiser les zones d'activation du modèle.
- **Analyse du Shortcut Learning** : Identification d'un biais où le modèle se focalisait sur des marqueurs radiologiques (lettres, bords) plutôt que sur le tissu pulmonaire.
- **Stratégie de Zoom/Crop** : Mise en place d'un pré-traitement de "Center Cropping" (zoom de 20% à 50%) pour forcer le modèle à analyser le centre de l'image.
- **Validation** : Vérification via Grad-CAM que l'attention du modèle est désormais correctement localisée sur les poumons.

*(... Les prochaines évolutions seront inscrites à la suite ...)*

### [07 Mai 2026] - Nettoyage et Refactorisation (Pre-GitHub Push)
- **Nettoyage de la racine** : Déplacement des scripts isolés (`scratch.py`, `patch_nb.py`, `preview_zoom.py`...) vers un dossier dédié `scripts/`.
- **Organisation des Modèles** : Migration des poids d'entraînement `.keras` volumineux vers le dossier standard `models/trained/` (protégé par le `.gitignore` pour éviter l'envoi de fichiers très lourds par accident).
- **Mise à plat des Notebooks** : Sortie de tous les fichiers `.ipynb` à la racine de `notebooks/` pour simplifier la structure, et suppression des dossiers vides obsolètes (`01_exploration`, `02_preprocessing`, `03_modeling`).

### [07 Mai 2026] - Synchronisation de la Documentation Technique
- **Audit de `PROJECT_EXPLANATION.md`** : Constat d'un fort décalage entre la théorie documentée et la pratique avancée du projet.
- **Réécriture de l'Architecture** : Ajout officiel à la documentation des nouvelles structures : Pipeline Hiérarchique (`hierarchical_pipeline.py`), l'évaluation (`evaluate_hierarchical.py`), et les modèles splittés (binaire vs multiclasses).
- **Valorisation du Explainable AI (XAI)** : Ajout de toute une section dédiée à l'interprétabilité avec le module `grad_cam.py`.
- **Nettoyage final** : Suppression de l'ancien `data_loader.py` obsolète au profit exclusif de la version robuste `data_loader_v3.py`.

### [07 Mai 2026] - Implémentation du Modèle Expert (Bactérie vs Virus)
- **Refonte de l'Étape 2 (Pipeline Hiérarchique)** : Abandon de l'ancien modèle multiclasses pour la seconde étape du diagnostic. Passage à une approche de classification purement binaire (0 = Bactérie, 1 = Virus) via Sigmoïd.
- **Data Loader Dynamique** : Création de `create_subtype_generators` dans `data_loader_v3.py` permettant d'ignorer le dossier "Normal" sans dupliquer de fichiers sur le disque dur.
- **Nouveau Script d'Entraînement** : Ajout de `train_subtype.py` à la racine pour permettre le fine-tuning d'un EfficientNetB0 exclusif aux poumons malades.

### [07 Mai 2026] - Clôture de la V1 Keras (Livrable d'Évaluation)
- **Création du Notebook Bilan** : Génération de `notebooks/05_hierarchical_evaluation.ipynb` regroupant la preuve de concept complète.
- **Démonstration Quantitative** : Intégration de l'évaluation hiérarchique avec calcul des Précisions, Rappels, F1-Scores et affichage d'une matrice de confusion globale à 3 classes.
- **Démonstration Qualitative (XAI)** : Application du `grad_cam.py` dans le notebook pour prouver visuellement que le sous-modèle spécialisé (Bactérie/Virus) analyse correctement le tissu pulmonaire (et évite le Shortcut Learning).
