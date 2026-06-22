# 🚀 QUICKSTART - Zoidberg

## ✅ Setup Terminé !

Ton environnement Python est maintenant **complètement configuré** et prêt à l'emploi !

---

## 📦 Packages Installés

- ✅ **TensorFlow 2.21.0** (avec Keras 3.14.1)
- ✅ **NumPy 2.1.0**
- ✅ **Pandas 3.0.3**
- ✅ **Scikit-learn 1.8.0**
- ✅ **Matplotlib 3.10.9**
- ✅ **Seaborn 0.13.2**
- ✅ **Pillow 11.0.0**

**Mode**: CPU uniquement (pas de GPU détecté)

---

## 🎮 Commandes Rapides

### Activer l'environnement
```bash
source venv/Scripts/activate  # Windows Git Bash
# OU
activate.bat                   # Windows CMD/PowerShell
# OU
source activate.sh             # Linux/macOS
```

### Désactiver
```bash
deactivate
```

### Tester l'installation
```bash
python test_environment.py      # Test des packages
python test_model_loading.py    # Test des modèles entraînés
```

---

## 🏃 Lancer les Entraînements

### 1️⃣ Entraînement Binaire (Normal vs Pneumonie)
```bash
python main.py
```

### 2️⃣ Entraînement Sous-type (Bactérie vs Virus)
```bash
python train_subtype.py
```

---

## 📊 Modèles Disponibles

Tu as **10 modèles pré-entraînés** dans `models/trained/` :

| Modèle | Taille | Description |
|--------|--------|-------------|
| `efficientnet_binary_stage1.keras` | 27 MB | Détection binaire (EfficientNet) |
| `efficientnet_multiclass_stage1.keras` | 34 MB | Classification 3 classes |
| `efficientnet_subtype_binary.keras` | 27 MB | Expert Bactérie/Virus |
| `zoidberg_cnn_best_crop_v1.h5` | 29 MB | CNN from scratch (crop) |
| + 6 autres modèles | - | Versions expérimentales |

**Status**: ✅ Tous les modèles se chargent correctement !

---

## 📓 Explorer avec Jupyter

```bash
pip install jupyter notebook
jupyter notebook
```

Notebooks disponibles dans `notebooks/` :
- `04_interpretability.ipynb` - Grad-CAM et explicabilité
- `05_hierarchical_evaluation.ipynb` - Pipeline hiérarchique
- `06_final_matrices.ipynb` - Métriques finales

---

## 🔄 Prochaines Étapes (Option B Complète)

### Phase 1 : Extraction des Métriques (Jour 1)
- [ ] Extraire toutes les métriques de tous les modèles
- [ ] Créer le tableau comparatif global
- [ ] Générer les courbes ROC comparatives
- [ ] Calculer les temps d'entraînement

### Phase 2 : Analyses Qualitatives (Jour 2)
- [ ] Error Analysis : identifier les cas mal classés
- [ ] Grad-CAM sur les erreurs
- [ ] Rédiger le papier final (8-10 pages)
- [ ] Créer 3-4 figures de synthèse

---

## 📝 Documentation Complète

- 📖 [README.md](README.md) - Vue d'ensemble du projet
- 🏛️ [ARCHITECTURE.md](ARCHITECTURE.md) - Choix architecturaux
- 🛠️ [SETUP.md](SETUP.md) - Guide d'installation détaillé
- 📊 [reports/final_report.md](reports/final_report.md) - Résultats finaux

---

## ⚠️ Notes Importantes

- Le dataset n'est **pas inclus** dans le repo (trop volumineux)
- Les modèles `.keras` sont **gitignorés** (sauf si tu veux les commiter)
- TensorFlow tourne en **mode CPU** sous Windows natif (pour GPU: utilise WSL2)
- Les warnings oneDNN sont normaux et n'affectent pas les résultats

---

## 🆘 Besoin d'Aide ?

```bash
# Réinstaller un package
pip install <package> --force-reinstall

# Vérifier les versions
pip list | grep tensorflow

# Nettoyer le cache
pip cache purge
```

---

**🎉 TON ENVIRONNEMENT EST PRÊT ! Tu peux maintenant passer à la phase d'analyse et de rédaction du papier final.**
