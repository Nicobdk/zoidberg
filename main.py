# from data_loader import load_data
# from data_exploration import explore_data
# from baseline_model import train_baseline_model
# from cnn_model import train_cnn_model
# from evaluation import evaluate_models

import json
import random
    
from data_loader import create_data_generator

from cnn_model import build_cnn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from data_exploration import visualize_batch, visualize_specific_class, show_class_distribution, plot_confusion_matrix, plot_roc_curve

DATA_DIR = "chest_Xray"

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

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

    plt.savefig("training_curves_v1.png")

    plt.show()


def main():
    train_gen, val_gen, test_gen = create_data_generator(DATA_DIR)

    model = build_cnn()

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
        patience=2,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        "zoidberg_binary_best_v1.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    y_prob=model.predict(test_gen)

    y_pred = (y_prob > 0.5).astype(int).flatten()

    y_true = test_gen.classes

    plot_confusion_matrix(y_true, y_pred)

    plot_roc_curve(y_true, y_prob)

    plot_history(history)

    model.evaluate(test_gen)

    report = classification_report(y_true, y_pred, output_dict=True)
    print(report)

    with open("classification_report_v1.json", "w") as f:
        json.dump(report, f)

    model.save("zoidberg_binary_v1.h5")

    with open("training_history_v1.json", "w") as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    main()
