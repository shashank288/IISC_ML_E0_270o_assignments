import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def preprocess_data(
       X: np.ndarray,
       medians: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to fill the missing values using median imputation.
    If medians is None, compute from X (fit). Otherwise use provided medians (transform).

    :param X: np.ndarray - input features (may contain NaN)
    :param medians: optional pre-computed medians from training data
    :return: Tuple of (X_filled, medians)
    """
    if medians is None:
        medians = np.nanmedian(X, axis=0)
    X_filled = X.copy()
    nan_indices = np.where(np.isnan(X_filled))
    X_filled[nan_indices] = np.take(medians, nan_indices[1])
    return X_filled, medians

def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from csv file and split into train and val set.
    Relabel the targets to M=1, B=0.

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (X_train, y_train, X_val, y_val)
    """
    # load data
    df = pd.read_csv(path)

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Relabel the targets to M=1, B=0
    y = (df['diagnosis'] == 'M').astype(int).values
    X = df.drop('diagnosis', axis=1).values.astype(float)

    # Stratified 70/30 split
    idx_class0 = np.where(y == 0)[0]
    idx_class1 = np.where(y == 1)[0]

    n_train_0 = int(0.7 * len(idx_class0))
    n_train_1 = int(0.7 * len(idx_class1))

    train_idx = np.concatenate([idx_class0[:n_train_0], idx_class1[:n_train_1]])
    val_idx = np.concatenate([idx_class0[n_train_0:], idx_class1[n_train_1:]])

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ===== Evaluation Metrics =====

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute 2x2 confusion matrix for binary classification.
    Returns [[TN, FP], [FN, TP]]
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute per-class precision, recall, F1, and macro/weighted averages.
    Returns dict with all metrics.
    """
    classes = np.unique(y_true)
    n_total = len(y_true)
    metrics = {}

    precisions, recalls, f1s, supports = [], [], [], []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    metrics['precision_per_class'] = precisions
    metrics['recall_per_class'] = recalls
    metrics['f1_per_class'] = f1s
    metrics['support_per_class'] = supports

    # Macro average (unweighted mean)
    metrics['precision_macro'] = np.mean(precisions)
    metrics['recall_macro'] = np.mean(recalls)
    metrics['f1_macro'] = np.mean(f1s)

    # Weighted average
    weights = supports / n_total
    metrics['precision_weighted'] = np.sum(precisions * weights)
    metrics['recall_weighted'] = np.sum(recalls * weights)
    metrics['f1_weighted'] = np.sum(f1s * weights)

    return metrics


def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute ROC curve (FPR, TPR) at various thresholds.
    y_scores: probability of positive class (class 1).
    Returns (fpr, tpr, thresholds)
    """
    # Sort by decreasing score
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]
    scores_sorted = y_scores[desc_idx]

    # Distinct thresholds
    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_indices = np.concatenate([distinct_indices, [len(y_sorted) - 1]])

    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    tps = np.cumsum(y_sorted)[threshold_indices]
    fps = (threshold_indices + 1) - tps

    tpr = np.concatenate([[0], tps / total_pos])
    fpr = np.concatenate([[0], fps / total_neg])
    thresholds = np.concatenate([[scores_sorted[0] + 1], scores_sorted[threshold_indices]])

    return fpr, tpr, thresholds


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve using trapezoidal rule.
    """
    fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
    return np.trapz(tpr, fpr)


def compute_pr_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute Precision-Recall curve at various thresholds.
    Returns (precision, recall, thresholds)
    """
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]
    scores_sorted = y_scores[desc_idx]

    tps = np.cumsum(y_sorted)
    total_pos = np.sum(y_true == 1)

    precision = tps / np.arange(1, len(y_sorted) + 1)
    recall = tps / total_pos

    # Add start point (recall=0, precision=1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    thresholds = scores_sorted

    return precision, recall, thresholds


# ===== Feature Engineering =====

# Indices of 9 highly correlated features to drop (from correlation analysis)
# area_mean(3), perimeter_mean(2), radius_worst(20), area_worst(23),
# perimeter_worst(22), texture_worst(21), concavity_mean(6), perimeter_se(12), area_se(13)
CORRELATED_FEATURE_INDICES = [2, 3, 6, 12, 13, 20, 21, 22, 23]


def remove_correlated_features(X, indices=None):
    """
    Remove highly correlated features by column index.
    Returns (X_reduced, kept_mask).
    """
    if indices is None:
        indices = CORRELATED_FEATURE_INDICES
    mask = np.ones(X.shape[1], dtype=bool)
    mask[indices] = False
    return X[:, mask], mask


def apply_yeo_johnson(X, lambdas=None):
    """
    Apply Yeo-Johnson power transform per feature.
    If lambdas is None, fit optimal lambda per feature (on training data).
    Returns (X_transformed, lambdas).
    """
    from scipy.stats import yeojohnson, yeojohnson_normmax
    X_transformed = np.zeros_like(X)
    if lambdas is None:
        lambdas = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            col = X[:, j]
            X_transformed[:, j], lambdas[j] = yeojohnson(col)
    else:
        for j in range(X.shape[1]):
            col = X[:, j]
            lam = lambdas[j]
            # Apply Yeo-Johnson with known lambda
            out = np.zeros_like(col)
            pos = col >= 0
            neg = ~pos
            if lam != 0:
                out[pos] = ((col[pos] + 1) ** lam - 1) / lam
            else:
                out[pos] = np.log(col[pos] + 1)
            if lam != 2:
                out[neg] = -((-col[neg] + 1) ** (2 - lam) - 1) / (2 - lam)
            else:
                out[neg] = -np.log(-col[neg] + 1)
            X_transformed[:, j] = out
    return X_transformed, lambdas


def winsorize(X, lower_pct=0.01, upper_pct=0.99, bounds=None):
    """
    Clip feature values to [lower_pct, upper_pct] percentiles.
    If bounds is None, compute from X (fit). Otherwise use provided bounds (transform).
    Returns (X_clipped, bounds).
    """
    X_clipped = X.copy()
    if bounds is None:
        bounds = np.zeros((2, X.shape[1]))
        bounds[0] = np.nanpercentile(X, lower_pct * 100, axis=0)
        bounds[1] = np.nanpercentile(X, upper_pct * 100, axis=0)
    for j in range(X.shape[1]):
        X_clipped[:, j] = np.clip(X_clipped[:, j], bounds[0, j], bounds[1, j])
    return X_clipped, bounds