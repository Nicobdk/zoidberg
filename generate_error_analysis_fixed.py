#!/usr/bin/env python
"""
Error Analysis avec Grad-CAM - Projet Zoidberg
Identifie les cas mal classs et gnre des visualisations Grad-CAM
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import random
from PIL import Image

# Configuration
DATA_DIR = Path("data/raw/chest_Xray/test")
MODELS_DIR = Path("models/trained")
OUTPUT_DIR = Path("reports/figures/error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Modles  analyser
MODEL_BINARY = MODELS_DIR / "efficientnet_binary_stage1.keras"
MODEL_MULTICLASS = MODELS_DIR / "efficientnet_multiclass_stage1.keras"

# Classes
CLASS_NAMES = ['Normal', 'Bactrie', 'Virus']
CLASS_NAMES_BINARY = ['Normal', 'Pneumonie']

# Paramtres
IMG_SIZE = (128, 128)
NUM_ERRORS_TO_SHOW = 8  # Nombre d'exemples d'erreurs  visualiser

print("=" * 80)
print(" ERROR ANALYSIS + GRAD-CAM - PROJET ZOIDBERG")
print("=" * 80)


def load_test_data(class_mapping='multiclass'):
    """Charge le dataset de test"""

    print("\n[1/6] Chargement des donnes de test...")

    images = []
    labels = []
    file_paths = []

    if class_mapping == 'multiclass':
        class_dirs = {
            '1_NORMAL': 0,
            '2_BACTERIA': 1,
            '3_VIRUS': 2
        }
    else:  # binary
        class_dirs = {
            '1_NORMAL': 0,
            '2_BACTERIA': 1,
            '3_VIRUS': 1
        }

    for class_dir, label in class_dirs.items():
        class_path = DATA_DIR / class_dir

        if not class_path.exists():
            print(f"  Dossier introuvable : {class_path}")
            continue

        class_files = list(class_path.glob("*.jpeg")) + list(class_path.glob("*.jpg"))
        print(f"   {class_dir:<15} : {len(class_files)} images")

        for img_path in class_files:
            try:
                # Charger l'image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img) / 255.0  # Normalisation

                images.append(img_array)
                labels.append(label)
                file_paths.append(str(img_path))
            except Exception as e:
                print(f"    Erreur chargement {img_path.name}: {e}")

    X = np.array(images)
    y = np.array(labels)

    print(f"\n   {len(X)} images charges")
    print(f"     Shape: {X.shape}")

    return X, y, file_paths


def find_misclassified(y_true, y_pred, y_proba, file_paths, top_n=10):
    """Identifie les cas mal classs avec les pires confidences"""

    print("\n[3/6] Identification des erreurs...")

    errors = []

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            # C'est une erreur
            confidence = y_proba[i][y_pred[i]]
            errors.append({
                'index': i,
                'true_label': y_true[i],
                'pred_label': y_pred[i],
                'confidence': confidence,
                'file_path': file_paths[i]
            })

    print(f"   Total erreurs : {len(errors)}/{len(y_true)} ({len(errors)/len(y_true)*100:.1f}%)")

    # Trier par confiance (pires erreurs = haute confiance mais mauvaise prdiction)
    errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)

    # Diversifier les types d'erreurs
    error_types = {}
    for err in errors_sorted:
        key = (err['true_label'], err['pred_label'])
        if key not in error_types:
            error_types[key] = []
        error_types[key].append(err)

    print(f"\n  Types d'erreurs dtectes :")
    for (true_l, pred_l), errs in error_types.items():
        true_name = CLASS_NAMES[true_l] if true_l < len(CLASS_NAMES) else f"Class {true_l}"
        pred_name = CLASS_NAMES[pred_l] if pred_l < len(CLASS_NAMES) else f"Class {pred_l}"
        print(f"     {true_name}  {pred_name} : {len(errs)} erreurs")

    # Slectionner top N en diversifiant
    selected_errors = []
    for error_type in error_types.values():
        selected_errors.extend(error_type[:2])  # 2 exemples par type
        if len(selected_errors) >= top_n:
            break

    return selected_errors[:top_n]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Gnre une heatmap Grad-CAM"""

    # Crer un modle qui mappe l'input  la couche conv + prdictions
    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Calcul du gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient de la classe prdite par rapport  la feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Moyenne des gradients (pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondration de la feature map
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisation
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def find_last_conv_layer(model):
    """Trouve la dernire couche de convolution"""

    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name

    # Si pas trouv, chercher dans les sous-modles
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if 'conv' in sublayer.name.lower():
                    return sublayer.name

    return None


def visualize_error_with_gradcam(img_array, error_info, model, class_names, save_path):
    """Visualise une erreur avec Grad-CAM"""

    # Trouver la dernire couche conv
    last_conv_layer = find_last_conv_layer(model)

    if last_conv_layer is None:
        print(f"    Pas de couche conv trouve dans le modle")
        # Visualisation sans Grad-CAM
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_array)
        ax.axis('off')
        ax.set_title(f"Vraie: {class_names[error_info['true_label']]} | "
                    f"Prdite: {class_names[error_info['pred_label']]} "
                    f"({error_info['confidence']*100:.1f}%)",
                    fontsize=10, fontweight='bold', color='red')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Gnrer Grad-CAM
    img_input = np.expand_dims(img_array, axis=0)
    heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer, error_info['pred_label'])

    # Crer la visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Image originale
    axes[0].imshow(img_array)
    axes[0].axis('off')
    axes[0].set_title('Image Originale', fontsize=11, fontweight='bold')

    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].axis('off')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=11, fontweight='bold')

    # Superposition
    axes[2].imshow(img_array)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[2].axis('off')
    axes[2].set_title('Superposition', fontsize=11, fontweight='bold')

    # Titre global
    true_name = class_names[error_info['true_label']]
    pred_name = class_names[error_info['pred_label']]
    confidence = error_info['confidence'] * 100

    fig.suptitle(f" ERREUR : Vraie Classe = {true_name} | Prdite = {pred_name} (Confiance: {confidence:.1f}%)",
                fontsize=13, fontweight='bold', color='#D32F2F')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"     {Path(save_path).name}")


def create_error_grid(X_test, errors, model, class_names, output_file):
    """Cre une grille de toutes les erreurs"""

    print(f"\n[5/6] Gnration de la grille d'erreurs...")

    n_errors = len(errors)
    n_cols = 4
    n_rows = (n_errors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_errors > 1 else [axes]

    for idx, error in enumerate(errors):
        if idx >= len(axes):
            break

        ax = axes[idx]
        img = X_test[error['index']]

        # Afficher l'image
        ax.imshow(img)
        ax.axis('off')

        # Titre avec info
        true_name = class_names[error['true_label']]
        pred_name = class_names[error['pred_label']]
        confidence = error['confidence'] * 100

        title_color = '#D32F2F' if confidence > 70 else '#F57C00'
        ax.set_title(f"Vraie: {true_name}\\nPrd: {pred_name} ({confidence:.0f}%)",
                    fontsize=9, fontweight='bold', color=title_color)

    # Cacher les axes non utiliss
    for idx in range(n_errors, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Exemples d\\'Erreurs de Classification ({n_errors} cas)',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"   Grille sauvegarde : {output_file}")
    plt.close()


def analyze_model(model_path, model_name, class_names, class_mapping='multiclass'):
    """Analyse complte d'un modle"""

    print(f"\n{'='*80}")
    print(f" ANALYSE DU MODLE : {model_name}")
    print(f"{'='*80}")

    # Charger le modle
    print(f"\n[2/6] Chargement du modle {model_name}...")
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"   Modle charg : {model_path.name}")
    except Exception as e:
        print(f"   Erreur chargement modle : {e}")
        return

    # Charger les donnes
    X_test, y_true, file_paths = load_test_data(class_mapping)

    # Prdictions
    print(f"\n[2/6] Prdictions sur le test set...")
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    accuracy = np.mean(y_pred == y_true) * 100
    print(f"   Accuracy : {accuracy:.2f}%")

    # Identifier les erreurs
    errors = find_misclassified(y_true, y_pred, y_proba, file_paths, top_n=NUM_ERRORS_TO_SHOW)

    if len(errors) == 0:
        print("   Aucune erreur trouve (modle parfait!)")
        return

    # Visualiser chaque erreur avec Grad-CAM
    print(f"\n[4/6] Gnration des Grad-CAM pour {len(errors)} erreurs...")

    for i, error in enumerate(errors, 1):
        img = X_test[error['index']]
        save_path = OUTPUT_DIR / f"{model_name}_error_{i:02d}.png"
        visualize_error_with_gradcam(img, error, model, class_names, save_path)

    # Crer la grille
    grid_file = OUTPUT_DIR / f"{model_name}_error_grid.png"
    create_error_grid(X_test, errors, model, class_names, grid_file)

    print(f"\n[6/6]  Analyse termine pour {model_name}")


if __name__ == "__main__":

    # Vrifier les modles
    if not MODEL_MULTICLASS.exists():
        print(f" Modle introuvable : {MODEL_MULTICLASS}")
        print("   Vrifier le chemin dans models/trained/")
    else:
        # Analyser le modle multi-classes
        analyze_model(
            MODEL_MULTICLASS,
            "EfficientNet_Multiclass",
            CLASS_NAMES,
            'multiclass'
        )

    print("\n" + "=" * 80)
    print(" ERROR ANALYSIS TERMINE !")
    print("=" * 80)
    print(f"\nFichiers gnrs dans : {OUTPUT_DIR}")
    print(f"   Erreurs individuelles avec Grad-CAM")
    print(f"   Grille rcapitulative")
    print("\n Ces visualisations montrent POURQUOI le modle se trompe")
    print("   et o il regarde (zones d'attention via Grad-CAM)")
