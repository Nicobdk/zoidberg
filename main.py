# from src.data.data_loader import load_data
# from src.visualization.data_exploration import explore_data
# from src.models.baseline_model import train_baseline_model
# from src.models.cnn_model import train_cnn_model
# from src.models.evaluation import evaluate_models

from src.visualization.data_exploration import plot_multiclass_roc
import json
import random

import pathlib
    
from src.data.data_loader import create_data_generator

from src.models.cnn_model import build_cnn
from src.models.transfert_model import build_efficientnet

from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.data_exploration import visualize_batch, visualize_specific_class, show_class_distribution, plot_confusion_matrix, plot_roc_curve

DATA_DIR = "data/raw/chest_Xray"
# dossiers = [DATA_DIR+"/train/1_NORMAL", DATA_DIR+'/train/2_BACTERIA', DATA_DIR+'/train/3_VIRUS'
#     , DATA_DIR+"/test/1_NORMAL", DATA_DIR+'/test/2_BACTERIA', DATA_DIR+'/test/3_VIRUS',
#     DATA_DIR+"/val/1_NORMAL", DATA_DIR+'/val/2_BACTERIA', DATA_DIR+'/val/3_VIRUS'
#     ]
# extensions_images = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

def verifier_et_compter():
    print(f"{'Dossier':<30} | {'Images trouvées':<15} | {'Statut'}")
    print("-" * 65)
    
    for d in dossiers:
        chemin = pathlib.Path(d)
        
        if not chemin.exists():
            print(f"{d:<30} | {'0':<15} | ❌ Dossier introuvable")
            continue
            
        # .rglob('*') permet de chercher dans TOUS les sous-dossiers
        nb_images = sum(1 for f in chemin.rglob('*') if f.suffix.lower() in extensions_images)
        
        print(f"{d:<30} | {nb_images:<15} | ✅ OK")

def plot_history(history):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.title("Loss")

    plt.savefig("reports/figures/training_curves_v1.png")

    plt.show()


def main():
    # L'ordre retourné par data_loader est : train_gen, test_gen, val_gen
    train_gen, test_gen, val_gen = create_data_generator(DATA_DIR)

    # model = build_cnn()
    model = build_efficientnet()

    classes = train_gen.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes
    )

    class_weights = dict(enumerate(class_weights))

    print(class_weights)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        "models/trained/zoidberg_cnn_best_crop_v1.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    y_true = test_gen.classes

    y_pred_probs=model.predict(test_gen)

    y_pred = np.argmax(y_pred_probs, axis=1)


    plot_confusion_matrix(y_true, y_pred)

    #Only banary classes
    plot_roc_curve(y_true, y_pred_probs)

    plot_multiclass_roc(y_true, y_pred_probs, n_classes=3)

    plot_history(history)

    model.evaluate(test_gen)

    report = classification_report(y_true, y_pred, output_dict=True)
    print(report)

    with open("models/evaluation/classification_report_v2.json", "w") as f:
        json.dump(report, f)

    ModelCheckpoint("models/trained/zoidberg_efficientnet_v3.keras", ...)

    with open("models/evaluation/training_history_v1.json", "w") as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    main()
    # print(f"Le script cherche à partir de : {pathlib.Path('.').absolute()}\n")
    # verifier_et_compter()
