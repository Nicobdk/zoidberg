# 🛠️ Guide d'Installation - Zoidberg

## 📦 Installation de l'Environnement

### 1️⃣ Cloner le projet
```bash
git clone <url-du-repo>
cd zoidberg
```

### 2️⃣ Créer l'environnement virtuel
```bash
python -m venv venv
```

### 3️⃣ Activer l'environnement

**Option Rapide (Scripts fournis):**
```bash
# Windows (PowerShell / CMD)
activate.bat

# Linux / macOS / Git Bash
source activate.sh
```

**Option Manuelle:**
```bash
# Windows (Git Bash / PowerShell)
source venv/Scripts/activate

# Linux / macOS
source venv/bin/activate
```

### 4️⃣ Mettre à jour pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

### 6️⃣ Vérifier l'installation
```bash
python test_environment.py
```

Tu devrais voir tous les packages avec un ✅.

---

## 🎮 Utilisation

### Activer l'environnement (à chaque session)
```bash
source venv/Scripts/activate  # Windows
source venv/bin/activate      # Linux/macOS
```

### Désactiver l'environnement
```bash
deactivate
```

---

## 📊 Lancer les Entraînements

### Entraînement Binaire (Normal vs Pneumonie)
```bash
python main.py
```

### Entraînement Sous-type (Bactérie vs Virus)
```bash
python train_subtype.py
```

### Explorer avec Jupyter
```bash
pip install jupyter notebook
jupyter notebook
```

---

## 🧪 Vérifications Importantes

### Vérifier TensorFlow
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
```

### Vérifier la structure des données
Le dataset doit être dans `data/raw/chest_Xray/` avec la structure :
```
data/raw/chest_Xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 🚨 Dépannage

### Erreur d'import TensorFlow
```bash
pip uninstall tensorflow
pip install tensorflow==2.21.0
```

### Erreur Keras
Keras est maintenant inclus dans TensorFlow 2.x, utilise :
```python
from tensorflow import keras
```

### Problème avec NumPy
```bash
pip install numpy==2.1.0 --force-reinstall
```

---

## 📝 Notes

- L'environnement virtuel (`venv/`) est **ignoré par git**
- Les modèles entraînés (`.h5`, `.keras`) sont dans `models/trained/` (ignorés par git)
- Le dataset n'est **pas inclus** dans le repo (trop volumineux)
