import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import seaborn as sns
import numpy as np

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Matrice de Confusion (Binaire)")
    plt.show()

def optimize_decision_threshold(y_true, y_pred_probs):
    """
    Trouve le seuil de décision (threshold) optimal pour la classification binaire.
    Optimise le F1-Score en testant différents seuils de 0.0 à 1.0.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculer le F1-score pour chaque seuil
    f1_scores = []
    
    for thresh in thresholds:
        predicted = (y_pred_probs > thresh).astype(int)
        score = f1_score(y_true, predicted)
        f1_scores.append(score)
        
    f1_scores = np.array(f1_scores)
    
    # Trouver l'indice du seuil maximum
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    best_f1 = f1_scores[optimal_idx]
    
    print(f"✅ Seuil optimal trouvé : {optimal_threshold:.4f}")
    print(f"📊 F1-Score optimal avec ce seuil : {best_f1:.4f}")
    
    # Affichage de la courbe ROC avec le point optimal
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], marker='o', color='red', s=100, label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR - Recall)')
    plt.title('Receiver Operating Characteristic avec Seuil Optimal')
    plt.legend(loc="lower right")
    plt.show()
    
    return optimal_threshold

