#!/usr/bin/env python
"""
Script de vérification de l'environnement Zoidberg
Vérifie que toutes les dépendances critiques sont installées
"""

import sys

def test_imports():
    """Test que tous les packages critiques peuvent être importés"""
    print("=" * 60)
    print("[TEST] ENVIRONNEMENT ZOIDBERG")
    print("=" * 60)

    packages = {
        'TensorFlow': 'tensorflow',
        'Keras': 'keras',
        'NumPy': 'numpy',
        'Pandas': 'pandas',
        'Scikit-learn': 'sklearn',
        'Matplotlib': 'matplotlib',
        'Seaborn': 'seaborn',
        'PIL (Pillow)': 'PIL',
    }

    results = []

    for name, module in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'N/A')
            print(f"[OK] {name:<20} {version}")
            results.append(True)
        except ImportError as e:
            print(f"[FAIL] {name:<20} MANQUANT - {e}")
            results.append(False)

    print("=" * 60)

    if all(results):
        print("[SUCCESS] Tous les packages sont installes !")

        # Test TensorFlow GPU (optionnel)
        try:
            import tensorflow as tf
            print(f"\n[INFO] TensorFlow Backend: {tf.config.list_physical_devices()}")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"[GPU] {len(gpus)} GPU(s) disponible(s)")
            else:
                print("[CPU] Mode CPU uniquement (pas de GPU detecte)")
        except Exception as e:
            print(f"[WARNING] Impossible de verifier le backend TF : {e}")

        return 0
    else:
        print("[FAIL] Certains packages manquent")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
