import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data.data_loader_v3 import create_subtype_generators
from src.models.subtype_model import build_subtype_efficientnet

# Paramètres
DATA_DIR = "data/raw/chest_Xray"
MODEL_SAVE_DIR = "models/trained"
MODEL_NAME = "efficientnet_subtype_binary.keras"
EPOCHS = 20

def train_subtype():
    print("🚀 Initialisation de l'entraînement du modèle Sous-Type (Bactérie vs Virus)")
    
    # 1. Chargement des données filtrées
    print("⏳ Chargement des données (NORMAL ignoré)...")
    train_gen, test_gen, val_gen = create_subtype_generators(DATA_DIR)
    
    # 2. Construction du modèle
    print("🛠️ Construction du modèle EfficientNet...")
    model = build_subtype_efficientnet()
    
    # 3. Callbacks (Sauvegarde, Early Stopping, Réduction du Learning Rate)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    ]
    
    # 4. Entraînement
    print("🔥 Début de l'entraînement...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print(f"✅ Entraînement terminé ! Meilleur modèle sauvegardé sous : {model_path}")

if __name__ == "__main__":
    train_subtype()
