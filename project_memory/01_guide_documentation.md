# 🧭 Guide de la Documentation - Projet Zoidberg

Ce document sert d'index central pour toute la documentation du projet. Il offre un récapitulatif global de l'avancement et répertorie tous les documents disponibles pour faciliter la navigation.

---

## 🌍 Récapitulatif Global du Projet

Le projet **Zoidberg** est une solution d'intelligence artificielle dédiée à l'analyse de radiographies thoraciques pour la détection de la pneumonie.

### 🎯 Objectifs
- **Classification Binaire** : Distinguer les poumons sains des poumons atteints de pneumonie (Réussi : Recall 98%).
- **Classification Multi-classes** : Identifier l'origine de la pneumonie : **Virale** ou **Bactérienne** (En cours d'optimisation : F1-Score ~56%).

### 🚀 Points Clés de l'Avancement
1.  **Modélisation Hybride** : Utilisation d'un CNN "from scratch" pour la baseline et d'EfficientNetB0 (Transfer Learning) pour la précision.
2.  **Interprétabilité (Grad-CAM)** : Validation visuelle des zones d'intérêt du modèle pour garantir une pertinence médicale.
3.  **Lutte contre les Biais** : Mise en place d'une stratégie de "Center Cropping" pour éliminer l'influence des marqueurs radiologiques (Shortcut Learning).
4.  **Architecture MLOps** : Structuration claire du code (`src/`), des données (`data/`) et du suivi (`project_memory/`).

---

## 🗺️ Cartographie de la Documentation

Voici la liste des documents disponibles dans le projet et leurs rôles respectifs :

### 📖 Documentation Générale
- **[README.md](../README.md)** : Vue d'ensemble du projet, installation, et résultats synthétiques.
- **[PROJECT_EXPLANATION.md](../PROJECT_EXPLANATION.md)** : Description détaillée de l'architecture logicielle et du rôle de chaque fichier Python.

### 🧠 Mémoire et Suivi (Project Memory)
- **[00_recap_etat_actuel.md](./00_recap_etat_actuel.md)** : État technique détaillé des modèles, des algorithmes de traitement et des métriques actuelles.
- **[CHANGELOG.md](./CHANGELOG.md)** : Journal de bord chronologique des modifications, ajouts de features et pivots stratégiques.
- **[01_guide_documentation.md](./01_guide_documentation.md)** : (Ce document) Index et récapitulatif global.

### 📊 Rapports et Analyses
- **[reports/final_report.md](../reports/final_report.md)** : Rapport final destiné à la présentation des résultats (En cours de rédaction).
- **[notebooks/](../notebooks/)** : Dossier contenant les analyses exploratoires, le prétraitement et les tests d'interprétabilité.

---

> Ce guide est maintenu à jour pour assurer la continuité du projet et faciliter l'onboarding de nouveaux contributeurs.
