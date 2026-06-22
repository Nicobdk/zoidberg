# 📑 Index de la Documentation - Projet Zoidberg

**Projet** : Détection de Pneumonie par Deep Learning  
**Auteur** : Nicolas BRODBECK  
**Date de mise à jour** : 01 Juin 2026

---

## 🎯 Documents Principaux

### 1. Pour une Vue Rapide
📄 **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - Résumé exécutif (2 pages)
- Vue d'ensemble du projet
- Résultats clés en un coup d'œil
- Recommandations et perspectives
- **⏱️ Lecture : 5 minutes**

---

### 2. Pour l'Analyse Détaillée
📊 **[COMPARATIVE_ANALYSIS.md](COMPARATIVE_ANALYSIS.md)** - Rapport comparatif complet
- Tableau comparatif de tous les modèles
- Métriques détaillées par modèle
- Analyse des forces et faiblesses
- Insights et recommandations
- **⏱️ Lecture : 15-20 minutes**

---

### 3. Pour les Résultats Finaux
📈 **[final_report.md](final_report.md)** - Rapport final Keras
- Évaluation du pipeline hiérarchique
- Performances de l'expert sous-type
- Conclusion et prochaines étapes (PyTorch)
- **⏱️ Lecture : 10 minutes**

---

## 📊 Données Exportées

### Fichiers CSV (pour Excel/traitement)
📁 **[models_comparison.csv](models_comparison.csv)** - Données tabulaires
- Import facile dans Excel/Google Sheets
- Toutes les métriques des 5 modèles
- Prêt pour graphiques personnalisés

---

## 📈 Visualisations Disponibles

### Dans `reports/figures/`

#### Graphiques Comparatifs
- 📊 **`roc_curves_comparative.png`** - Courbes ROC comparatives (modèles binaires)
- 📊 **`auc_comparison_bar.png`** - Comparaison des scores AUC en barres
- 📊 **`accuracy_comparison_all_models.png`** - Accuracy de tous les modèles

#### Résultats Individuels
- 📉 **`training_curves_v1.png`** - Courbes d'apprentissage (Loss/Accuracy)
- 🎯 **`hierarchical_confusion_matrix.png`** - Matrice de confusion du pipeline

#### Explicabilité (Grad-CAM)
- 👁️ **`grad_cam_bacteria_example.png`** - Exemple de heatmap d'attention
- 🔍 **`zoom_preview.png`** - Prévisualisation des transformations

---

## 🗂️ Organisation Complète du Projet

```
zoidberg/
│
├── reports/                              ← VOUS ÊTES ICI
│   ├── INDEX.md                          ← Ce fichier
│   ├── EXECUTIVE_SUMMARY.md              ← Résumé exécutif
│   ├── COMPARATIVE_ANALYSIS.md           ← Analyse comparative
│   ├── final_report.md                   ← Rapport final
│   ├── models_comparison.csv             ← Données CSV
│   └── figures/                          ← Toutes les visualisations
│       ├── roc_curves_comparative.png
│       ├── auc_comparison_bar.png
│       ├── accuracy_comparison_all_models.png
│       ├── hierarchical_confusion_matrix.png
│       ├── grad_cam_bacteria_example.png
│       └── training_curves_v1.png
│
├── README.md                             ← Vue d'ensemble du projet
├── ARCHITECTURE.md                       ← Réflexion architecturale
├── PROJECT_EXPLANATION.md                ← Documentation technique
├── SETUP.md                              ← Guide d'installation
├── QUICKSTART.md                         ← Démarrage rapide
│
├── models/
│   ├── trained/                          ← Modèles entraînés (.keras, .h5)
│   └── evaluation/                       ← Métriques JSON/TXT
│
├── notebooks/                            ← Jupyter notebooks d'exploration
│   ├── 04_interpretability.ipynb
│   ├── 05_hierarchical_evaluation.ipynb
│   └── 06_final_matrices.ipynb
│
├── src/                                  ← Code source
│   ├── data/                             ← Data loaders
│   ├── models/                           ← Architectures
│   └── visualization/                    ← Grad-CAM, plots
│
├── project_memory/                       ← Historique du projet
│   ├── 00_recap_etat_actuel.md
│   ├── 01_guide_documentation.md
│   └── CHANGELOG.md
│
└── scripts/                              ← Utilitaires
    ├── generate_comparative_report.py    ← Génère COMPARATIVE_ANALYSIS.md
    ├── generate_roc_curves.py            ← Génère graphiques ROC
    ├── test_environment.py               ← Test packages Python
    └── test_model_loading.py             ← Test chargement modèles
```

---

## 🚀 Utilisation Recommandée

### Pour une Présentation Orale (15 min)
1. **Slide 1-2** : EXECUTIVE_SUMMARY.md (Intro + Résultats)
2. **Slide 3-4** : Tableau comparatif de COMPARATIVE_ANALYSIS.md
3. **Slide 5-6** : Graphiques (ROC curves, Accuracy comparison)
4. **Slide 7** : Grad-CAM (Explicabilité)
5. **Slide 8** : Conclusions et perspectives

### Pour un Papier Écrit (8-10 pages)
**Structure suggérée** :
1. **Introduction** (1 page) - Contexte médical, objectifs
2. **Matériel & Méthodes** (2 pages) - Dataset, architectures
3. **Résultats** (2-3 pages) - Tableau + graphiques de COMPARATIVE_ANALYSIS.md
4. **Discussion** (2 pages) - Analyse des résultats, limites
5. **Conclusion** (0.5 page) - Synthèse et perspectives

**Sources à citer** :
- COMPARATIVE_ANALYSIS.md (tableau comparatif)
- final_report.md (résultats hiérarchiques)
- Figures du dossier `figures/`

### Pour une Démo Live
1. Activer l'environnement : `source venv/Scripts/activate`
2. Lancer le test : `python test_model_loading.py`
3. Ouvrir notebook : `jupyter notebook notebooks/05_hierarchical_evaluation.ipynb`
4. Montrer Grad-CAM en direct

---

## 📝 Génération des Rapports

### Régénérer le rapport comparatif
```bash
python generate_comparative_report.py
```
**Génère** :
- `reports/COMPARATIVE_ANALYSIS.md`
- `reports/models_comparison.csv`

### Régénérer les graphiques
```bash
python generate_roc_curves.py
```
**Génère** :
- `reports/figures/roc_curves_comparative.png`
- `reports/figures/auc_comparison_bar.png`
- `reports/figures/accuracy_comparison_all_models.png`

---

## 🎯 Checklist Rendu Final

### Documents Obligatoires
- [x] Rapport technique complet (COMPARATIVE_ANALYSIS.md)
- [x] Résumé exécutif (EXECUTIVE_SUMMARY.md)
- [x] Tableau comparatif des modèles (inclus dans rapports)
- [x] Visualisations (ROC, accuracy, Grad-CAM)

### Code & Reproductibilité
- [x] Code source structuré (`src/`)
- [x] Requirements.txt à jour
- [x] Scripts de test fonctionnels
- [x] Notebooks d'exploration

### Documentation
- [x] README.md complet
- [x] Guide d'installation (SETUP.md)
- [x] Architecture documentée (ARCHITECTURE.md)
- [x] Quickstart (QUICKSTART.md)

### Résultats
- [x] Modèles entraînés sauvegardés
- [x] Métriques JSON/CSV exportées
- [x] Matrices de confusion
- [x] Courbes d'apprentissage

---

## 💡 Conseils pour la Présentation

### Points à Mettre en Avant
✅ **Approche rigoureuse** : Baseline → CNN → Transfer → Hiérarchique  
✅ **Innovation** : Pipeline hiérarchique (+15% accuracy)  
✅ **Explicabilité** : Grad-CAM pour validation médicale  
✅ **Performances** : AUC 0.9897, Recall 98%  

### Points à Assumer/Expliquer
⚠️ **Limites virus/bactérie** : Limite médicale connue (radiographie seule insuffisante)  
⚠️ **Trade-off Recall/Precision** : Choix médical assumé (ne pas rater un malade)  
⚠️ **Déséquilibre classes** : Classe virus minoritaire → sur-prédiction bactérie  

---

## 🆘 Support

**Questions sur le code ?**  
→ Voir `PROJECT_EXPLANATION.md` et `ARCHITECTURE.md`

**Problèmes d'installation ?**  
→ Voir `SETUP.md` et `QUICKSTART.md`

**Besoin de métriques supplémentaires ?**  
→ Check `models/evaluation/` et notebooks Jupyter

---

**🎉 Tout est prêt pour ton rendu final !**

*Index généré le 01/06/2026 - Projet Zoidberg*
