# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def create_multiclass_generators(data_dir):

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    test_gen = test_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, test_gen, val_gen

def create_subtype_generators(data_dir):
    """
    Générateur spécifique pour le sous-modèle : ignore le dossier '1_NORMAL'
    et ne charge que '2_BACTERIA' et '3_VIRUS' en mode binaire.
    """
    subtype_classes = ["2_BACTERIA", "3_VIRUS"]

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15, # Un peu plus de rotation pour compenser la perte de data
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=subtype_classes,
        class_mode="binary", # 0 = BACTERIA, 1 = VIRUS
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=subtype_classes,
        class_mode="binary",
        subset="validation"
    )

    test_gen = test_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=subtype_classes,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, test_gen, val_gen