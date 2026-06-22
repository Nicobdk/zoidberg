from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

def build_subtype_efficientnet(input_shape=(224, 224, 3)):
    """
    Modèle spécialisé pour différencier Bactérie vs Virus.
    Sortie binaire (Sigmoid).
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # On gèle la majorité du modèle pour ne fine-tuner que les dernières couches
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        # Un seul neurone car classification binaire (0 = Bactérie, 1 = Virus)
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    model = build_subtype_efficientnet()
    model.summary()
