from keras import layers, models
from keras.applications import EfficientNetB0
from keras.optimizers import Adam

def build_binary_efficientnet():

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    for layer in base_model.layers[:-20]:
        layer.trainable=False

    model.compile(
        optimizer=Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model