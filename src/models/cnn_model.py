from keras import layers, models

def build_cnn(input_shape=(128, 128, 3)):
    model = models.Sequential([

        # ---- Entrée propre ----
        layers.Input(shape=input_shape),

        # ---- Bloc 1 ----
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # ---- Bloc 2 ----
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # ---- Bloc 3 (optionnel mais utile) ----
        # layers.Conv2D(128, (3, 3), padding="same"),
        # layers.BatchNormalization(),
        # layers.Activation("relu"),
        # layers.MaxPooling2D(2, 2),

        # ---- Régularisation ----
        layers.Dropout(0.5),

        # ---- Classification ----
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
