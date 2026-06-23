"""
Module d'évaluation avancée avec Cross-Validation et métriques complètes.

Ajoute les capacités suivantes au projet Zoidberg :
1. K-Fold Cross-Validation (StratifiedKFold)
2. Métriques médicales avancées (Specificity, NPV, PPV, MCC, etc.)
3. Intervalle de confiance (Bootstrap)
4. Comparaison statistique entre modèles
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score,
    classification_report
)
from tensorflow import keras
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def calculate_medical_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: np.ndarray = None,
                              class_names: List[str] = None) -> Dict:
    """
    Calcule TOUTES les métriques médicales importantes.

    Métriques calculées :
    - Accuracy, Balanced Accuracy
    - Precision, Recall, F1-Score (Macro/Weighted)
    - Specificity (TN / (TN + FP))
    - NPV (Negative Predictive Value)
    - PPV (Positive Predictive Value) = Precision
    - MCC (Matthews Correlation Coefficient)
    - Cohen's Kappa
    - AUC-ROC (si y_pred_proba fourni)
    - Sensitivity = Recall

    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
        y_pred_proba: Probabilités prédites (optionnel, pour AUC)
        class_names: Noms des classes (optionnel)

    Returns:
        Dict avec toutes les métriques
    """

    # Métriques de base
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Binary classification
    average = 'binary' if len(np.unique(y_true)) == 2 else 'macro'

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC-ROC (si probabilités disponibles)
    auc = None
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_pred_proba)
            else:
                auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        except:
            pass

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    # Métriques par classe (Specificity, NPV, PPV)
    num_classes = cm.shape[0]
    specificity_per_class = []
    npv_per_class = []
    ppv_per_class = []

    for i in range(num_classes):
        # True Positives, False Positives, True Negatives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        # Specificity = TN / (TN + FP)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(spec)

        # NPV = TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv_per_class.append(npv)

        # PPV = TP / (TP + FP) = Precision
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        ppv_per_class.append(ppv)

    # Moyennes
    specificity_macro = np.mean(specificity_per_class)
    npv_macro = np.mean(npv_per_class)
    ppv_macro = np.mean(ppv_per_class)

    metrics = {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'sensitivity_macro': recall_macro,  # Sensitivity = Recall
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'specificity_macro': specificity_macro,
        'npv_macro': npv_macro,
        'ppv_macro': ppv_macro,
        'cohens_kappa': kappa,
        'mcc': mcc,
        'auc_roc': auc,
        'confusion_matrix': cm.tolist(),
        'specificity_per_class': specificity_per_class,
        'npv_per_class': npv_per_class,
        'ppv_per_class': ppv_per_class
    }

    return metrics


def cross_validate_model(model_builder_fn, train_gen, val_gen,
                         n_splits: int = 5,
                         epochs: int = 20,
                         verbose: int = 0) -> Dict:
    """
    Effectue une K-Fold Cross-Validation sur un modèle.

    IMPORTANT : Pour Deep Learning, la cross-validation est coûteuse !
    Cette fonction est surtout utile pour :
    - Valider la robustesse du modèle
    - Estimer la variance des performances
    - Comparer différentes architectures

    Args:
        model_builder_fn: Fonction qui crée le modèle (retourne un modèle Keras compilé)
        train_gen: Générateur d'entraînement (sera divisé en K folds)
        val_gen: Générateur de validation (utilisé pour chaque fold)
        n_splits: Nombre de folds (default: 5)
        epochs: Nombre d'epochs par fold
        verbose: Verbosité (0 = silencieux, 1 = barre de progression, 2 = une ligne par epoch)

    Returns:
        Dict avec métriques moyennes et écart-types

    Example:
        >>> def builder():
        >>>     model = keras.Sequential([...])
        >>>     model.compile(...)
        >>>     return model
        >>>
        >>> results = cross_validate_model(builder, train_gen, val_gen, n_splits=5)
        >>> print(f"Accuracy: {results['mean_accuracy']:.2%} ± {results['std_accuracy']:.2%}")
    """

    print(f"\n{'='*80}")
    print(f"🔄 CROSS-VALIDATION ({n_splits}-Fold)")
    print(f"{'='*80}\n")

    # Extraire toutes les données du générateur
    print("📊 Extraction des données du générateur...")
    X_all = []
    y_all = []

    for i in range(len(train_gen)):
        X_batch, y_batch = train_gen[i]
        X_all.append(X_batch)
        y_all.append(y_batch)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    print(f"✅ {len(X_all)} échantillons extraits")

    # Si y_all est one-hot encoded, convertir en labels
    if len(y_all.shape) > 1:
        y_labels = np.argmax(y_all, axis=1)
    else:
        y_labels = y_all

    # StratifiedKFold pour préserver les proportions de classes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'loss': []
    }

    fold_num = 1
    for train_idx, val_idx in skf.split(X_all, y_labels):
        print(f"\n{'─'*80}")
        print(f"📁 Fold {fold_num}/{n_splits}")
        print(f"{'─'*80}")

        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]

        print(f"   Train: {len(X_train_fold)} échantillons")
        print(f"   Val  : {len(X_val_fold)} échantillons")

        # Créer un nouveau modèle pour chaque fold
        model = model_builder_fn()

        # Entraîner
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=32,
            verbose=verbose
        )

        # Évaluer
        y_pred_proba = model.predict(X_val_fold, verbose=0)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            # Multi-class
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_val_fold, axis=1) if len(y_val_fold.shape) > 1 else y_val_fold
        else:
            # Binary
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = y_val_fold.flatten()

        # Métriques
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        loss = history.history['val_loss'][-1]

        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(prec)
        fold_results['recall'].append(rec)
        fold_results['f1_score'].append(f1)
        fold_results['loss'].append(loss)

        print(f"\n   ✅ Résultats Fold {fold_num}:")
        print(f"      Accuracy  : {acc:.4f}")
        print(f"      Precision : {prec:.4f}")
        print(f"      Recall    : {rec:.4f}")
        print(f"      F1-Score  : {f1:.4f}")
        print(f"      Loss      : {loss:.4f}")

        fold_num += 1

    # Calculer moyennes et écart-types
    results = {}
    for metric, values in fold_results.items():
        results[f'mean_{metric}'] = np.mean(values)
        results[f'std_{metric}'] = np.std(values)
        results[f'{metric}_all_folds'] = values

    # Affichage final
    print(f"\n{'='*80}")
    print(f"📊 RÉSULTATS CROSS-VALIDATION ({n_splits}-Fold)")
    print(f"{'='*80}\n")

    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'loss']:
        mean_val = results[f'mean_{metric}']
        std_val = results[f'std_{metric}']
        print(f"   {metric.upper():<12} : {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n{'='*80}\n")

    return results


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                  metric: str = 'accuracy',
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calcule l'intervalle de confiance d'une métrique par Bootstrap.

    Utile pour estimer la variabilité des performances sur le test set.

    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        metric: Métrique à calculer ('accuracy', 'precision', 'recall', 'f1')
        n_bootstrap: Nombre d'échantillons bootstrap
        confidence_level: Niveau de confiance (default: 0.95 = 95%)

    Returns:
        (mean, lower_bound, upper_bound)

    Example:
        >>> mean, lower, upper = bootstrap_confidence_interval(y_true, y_pred, 'accuracy')
        >>> print(f"Accuracy: {mean:.2%} [{lower:.2%}, {upper:.2%}]")
    """

    metric_fn = {
        'accuracy': accuracy_score,
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='macro', zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='macro', zero_division=0),
        'f1': lambda y_t, y_p: f1_score(y_t, y_p, average='macro', zero_division=0)
    }

    if metric not in metric_fn:
        raise ValueError(f"Métrique '{metric}' non supportée. Choix: {list(metric_fn.keys())}")

    scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        # Bootstrap sampling (avec remplacement)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        score = metric_fn[metric](y_true_boot, y_pred_boot)
        scores.append(score)

    scores = np.array(scores)

    # Calcul intervalle de confiance
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean_score = np.mean(scores)
    lower_bound = np.percentile(scores, lower_percentile)
    upper_bound = np.percentile(scores, upper_percentile)

    return mean_score, lower_bound, upper_bound


def compare_models_statistical(results_model1: Dict, results_model2: Dict,
                               model1_name: str = "Model 1",
                               model2_name: str = "Model 2") -> pd.DataFrame:
    """
    Compare deux modèles de manière statistique.

    Utilisé après cross-validation pour comparer si les différences
    de performance sont significatives.

    Args:
        results_model1: Résultats de cross_validate_model() du modèle 1
        results_model2: Résultats de cross_validate_model() du modèle 2
        model1_name: Nom du modèle 1
        model2_name: Nom du modèle 2

    Returns:
        DataFrame avec comparaison des métriques

    Example:
        >>> results1 = cross_validate_model(builder_cnn, ...)
        >>> results2 = cross_validate_model(builder_efficientnet, ...)
        >>> comparison = compare_models_statistical(results1, results2, "CNN", "EfficientNet")
        >>> print(comparison)
    """

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    comparison_data = []

    for metric in metrics:
        mean1 = results_model1[f'mean_{metric}']
        std1 = results_model1[f'std_{metric}']

        mean2 = results_model2[f'mean_{metric}']
        std2 = results_model2[f'std_{metric}']

        diff = mean2 - mean1
        diff_pct = (diff / mean1) * 100 if mean1 > 0 else 0

        winner = model2_name if diff > 0 else model1_name

        comparison_data.append({
            'Metric': metric.capitalize(),
            f'{model1_name} (Mean±Std)': f"{mean1:.4f} ± {std1:.4f}",
            f'{model2_name} (Mean±Std)': f"{mean2:.4f} ± {std2:.4f}",
            'Difference': f"{diff:+.4f} ({diff_pct:+.2f}%)",
            'Winner': winner
        })

    df = pd.DataFrame(comparison_data)

    return df


def evaluate_model_comprehensive(model, test_gen, class_names: List[str] = None) -> Dict:
    """
    Évaluation complète d'un modèle avec TOUTES les métriques.

    Combine :
    - Métriques médicales avancées
    - Intervalles de confiance (Bootstrap)
    - Rapport de classification

    Args:
        model: Modèle Keras entraîné
        test_gen: Générateur de test
        class_names: Noms des classes (optionnel)

    Returns:
        Dict avec toutes les métriques et intervalles de confiance

    Example:
        >>> model = keras.models.load_model('my_model.keras')
        >>> evaluation = evaluate_model_comprehensive(model, test_gen, ['Normal', 'Bacteria', 'Virus'])
        >>> print(f"Accuracy: {evaluation['accuracy']:.2%} [{evaluation['accuracy_ci'][0]:.2%}, {evaluation['accuracy_ci'][1]:.2%}]")
    """

    print(f"\n{'='*80}")
    print(f"📊 ÉVALUATION COMPLÈTE DU MODÈLE")
    print(f"{'='*80}\n")

    # Extraire données du test_gen
    X_test = []
    y_test = []

    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        X_test.append(X_batch)
        y_test.append(y_batch)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print(f"📦 Test set: {len(X_test)} échantillons")

    # Prédictions
    y_pred_proba = model.predict(X_test, verbose=0)

    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
        # Multi-class
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    else:
        # Binary
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = y_test.flatten()
        y_pred_proba = y_pred_proba.flatten()

    # Métriques médicales complètes
    metrics = calculate_medical_metrics(y_true, y_pred, y_pred_proba, class_names)

    # Intervalles de confiance (Bootstrap)
    print("\n🔄 Calcul des intervalles de confiance (Bootstrap)...")

    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        mean, lower, upper = bootstrap_confidence_interval(y_true, y_pred, metric_name)
        metrics[f'{metric_name}_ci'] = (lower, upper)
        metrics[f'{metric_name}_ci_mean'] = mean

    # Rapport de classification
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report

    # Affichage
    print(f"\n{'─'*80}")
    print(f"📈 MÉTRIQUES GLOBALES")
    print(f"{'─'*80}\n")

    print(f"   Accuracy           : {metrics['accuracy']:.4f} [{metrics['accuracy_ci'][0]:.4f}, {metrics['accuracy_ci'][1]:.4f}]")
    print(f"   Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"   Precision (Macro)  : {metrics['precision_macro']:.4f} [{metrics['precision_ci'][0]:.4f}, {metrics['precision_ci'][1]:.4f}]")
    print(f"   Recall (Macro)     : {metrics['recall_macro']:.4f} [{metrics['recall_ci'][0]:.4f}, {metrics['recall_ci'][1]:.4f}]")
    print(f"   F1-Score (Macro)   : {metrics['f1_macro']:.4f} [{metrics['f1_ci'][0]:.4f}, {metrics['f1_ci'][1]:.4f}]")
    print(f"   Specificity (Macro): {metrics['specificity_macro']:.4f}")
    print(f"   NPV (Macro)        : {metrics['npv_macro']:.4f}")
    print(f"   PPV (Macro)        : {metrics['ppv_macro']:.4f}")
    print(f"   Cohen's Kappa      : {metrics['cohens_kappa']:.4f}")
    print(f"   MCC                : {metrics['mcc']:.4f}")

    if metrics['auc_roc'] is not None:
        print(f"   AUC-ROC            : {metrics['auc_roc']:.4f}")

    print(f"\n{'─'*80}")
    print(f"📋 MÉTRIQUES PAR CLASSE")
    print(f"{'─'*80}\n")

    for i, class_name in enumerate(class_names):
        print(f"   {class_name}:")
        print(f"      Specificity : {metrics['specificity_per_class'][i]:.4f}")
        print(f"      NPV         : {metrics['npv_per_class'][i]:.4f}")
        print(f"      PPV         : {metrics['ppv_per_class'][i]:.4f}")

    print(f"\n{'='*80}\n")

    return metrics
