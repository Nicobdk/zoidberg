import os
import pathlib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from src.models.hierarchical_pipeline import HierarchicalClassifier

DATA_DIR = "data/raw/chest_Xray/test"
BINARY_MODEL_PATH = "models/trained/efficientnet_binary_stage1.keras"
SUBTYPE_MODEL_PATH = "models/trained/efficientnet_subtype_binary.keras"

# Note: Pense à ajuster img_size selon le modèle que tu as entraîné (224 ou 128)
IMG_SIZE = (224, 224)

def evaluate_hierarchical_model(binary_model, subtype_model):
    print("⏳ Chargement des modèles hiérarchiques...")
    try:
        classifier = HierarchicalClassifier(
            binary_model_path=binary_model,
            subtype_model_path=subtype_model,
            img_size=IMG_SIZE
        )
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles. Vérifie les chemins : {e}")
        return

    print("✅ Modèles chargés avec succès. Début de l'évaluation sur le test set...")

    y_true = []
    y_pred = []
    
    # Mapping des répertoires aux vraies classes
    class_mapping = {
        "1_NORMAL": "NORMAL",
        "2_BACTERIA": "BACTERIA",
        "3_VIRUS": "VIRUS"
    }
    
    # Extensions valides
    valid_extensions = {".jpg", ".jpeg", ".png"}

    for folder_name, true_label in class_mapping.items():
        folder_path = pathlib.Path(DATA_DIR) / folder_name
        
        if not folder_path.exists():
            print(f"⚠️ Dossier introuvable: {folder_path} - On ignore cette classe.")
            continue
            
        images = [f for f in folder_path.glob("*") if f.suffix.lower() in valid_extensions]
        
        print(f"\nÉvaluation de la classe : {true_label} ({len(images)} images)")
        
        for i, img_path in enumerate(images):
            if i % 50 == 0:
                print(f"  [Progression] {i}/{len(images)} images analysées...")
            try:
                # Prédiction via le pipeline hiérarchique
                result = classifier.predict(str(img_path))
                predicted_label = result["final_prediction"]
                
                y_true.append(true_label)
                y_pred.append(predicted_label)
            except Exception as e:
                print(f"⚠️ Erreur avec l'image {img_path}: {e}")

    # Calcul des métriques globales
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f"🌟 ACCURACY GLOBALE (Pipeline Hiérarchique) : {acc:.2f} ({acc*100:.2f}%)")
    print("="*50)

    print("\n📊 RAPPORT DE CLASSIFICATION MÉDICAL :")
    print(classification_report(y_true, y_pred, target_names=["BACTERIA", "NORMAL", "VIRUS"]))

    # Matrice de confusion
    print("\n🖼️ Génération de la Matrice de Confusion...")
    cm = confusion_matrix(y_true, y_pred, labels=["NORMAL", "BACTERIA", "VIRUS"])
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Bactérie", "Virus"],
                yticklabels=["Normal", "Bactérie", "Virus"])
    plt.xlabel("Prédiction (Modèle Hiérarchique)")
    plt.ylabel("Réalité (True Label)")
    plt.title("Matrice de Confusion: Pipeline Hiérarchique (3 Classes)")
    
    save_path = "reports/figures/hierarchical_confusion_matrix.png"
    # S'assurer que le dossier existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ Matrice de confusion sauvegardée dans : {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_hierarchical_model(BINARY_MODEL_PATH, SUBTYPE_MODEL_PATH)
