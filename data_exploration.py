import matplotlib.pyplot as plt 
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns



def show_class_distribution(generator):
    labels = generator.classes
    classes, counts = np.unique(labels, return_counts=True)

    plt.bar(classes, counts)
    plt.xticks(classes, ['Normal', "Pneumonia"])
    plt.title("Classe distribution")
    plt.show()

def show_sample_images(generator, n=5):
    images, labels = next(generator)

    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title("Pneumonia" if labels[i] == 1 else "Normal")
        plt.axis("off")

    plt.show()

def visualize_batch(generator, n=6):
    images, labels = next(generator)

    plt.figure(figsize=(12, 4))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title("Pneumonia" if labels[i] == 1 else "Normal")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_class_distribution(generator):
    labels = generator.classes
    unique, counts = np.unique(labels, return_counts=True)

    plt.bar(["Normal", "Pneumonia"], counts)
    plt.title("Class Distribution")
    plt.show()

    print("Normal:", counts[0])
    print("Pneumonia:", counts[1])

def visualize_specific_class(generator, target_class=1, n=6):
    images, labels = next(generator)
    
    plt.figure(figsize=(12, 4))
    count = 0
    
    for i in range(len(labels)):
        if labels[i] == target_class:
            plt.subplot(1, n, count + 1)
            plt.imshow(images[i])
            plt.title("Pneumonia")
            plt.axis("off")
            count += 1
        if count == n:
            break

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    with open("auc_v1.txt", "w") as f:
        f.write(f"AUC: {roc_auc}")

    print(f"\nAUC Score: {roc_auc:.3f}")