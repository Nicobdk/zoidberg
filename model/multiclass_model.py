from keras.applications import EfficientNetB0
from keras import layers, models
from keras.optimizers import Adam


def build_efficientnet_multiclass(
    num_classes=3,
    input_shape=(224, 224, 3),
    fine_tune_layers=40,
    learning_rate=1e-6
):
    """
    Build EfficientNetB0 model for multi-class classification.
    """

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all layers first
    base_model.trainable = True

    # Freeze early layers
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def build_efficientnet_head_only():

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model