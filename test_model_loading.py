#!/usr/bin/env python
"""
Test de chargement des modeles Zoidberg
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("[TEST] CHARGEMENT DES MODELES ZOIDBERG")
print("=" * 60)

models_dir = "models/trained"
test_models = [
    "efficientnet_binary_stage1.keras",
    "efficientnet_multiclass_stage1.keras",
    "efficientnet_subtype_binary.keras"
]

success_count = 0
total_count = 0

for model_name in test_models:
    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        print(f"[SKIP] {model_name:<40} (fichier absent)")
        continue

    total_count += 1

    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"[OK]   {model_name:<40} ({len(model.layers)} couches)")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] {model_name:<40} Erreur: {str(e)[:40]}")

print("=" * 60)
print(f"[RESULT] {success_count}/{total_count} modeles charges avec succes")

if success_count == total_count and total_count > 0:
    print("[SUCCESS] Tous les modeles sont chargeables !")
    exit(0)
else:
    print("[FAIL] Certains modeles n'ont pas pu etre charges")
    exit(1)
