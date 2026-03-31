import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from model import *


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Run full evaluation on a dataset and print results.
    Returns (y_pred, y_proba, metrics_dict).
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # P(Malignant)

    acc = np.mean(y_pred == y)
    cm = compute_confusion_matrix(y, y_pred)
    metrics = compute_precision_recall_f1(y, y_pred)
    roc_auc = compute_roc_auc(y, y_proba)

    print(f"\n{dataset_name} Evaluation")
    print(f"Accuracy:{acc:.4f}")
    print(f"Precision (macro):{metrics['precision_macro']:.4f}")
    print(f"Recall (macro):{metrics['recall_macro']:.4f}")
    print(f"F1 (macro):{metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):{metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC:{roc_auc:.4f}")
    print(f"Confusion Matrix:")
    print(f"TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    return y_pred, y_proba, {**metrics, 'accuracy': acc, 'roc_auc': roc_auc, 'confusion_matrix': cm}


def find_threshold_for_recall(y_true, y_proba, target_recall=0.95):
    """
    Find the highest classification threshold that achieves at least target_recall.
    Uses actual probability values as candidate thresholds.
    Returns (threshold, precision_at_threshold, recall_at_threshold, f1_at_threshold).
    """
    # Sort samples by descending probability
    desc_idx = np.argsort(-y_proba)
    sorted_proba = y_proba[desc_idx]
    sorted_true = y_true[desc_idx]

    total_pos = np.sum(y_true == 1)
    required_tp = int(np.ceil(target_recall * total_pos))

    # Walk through sorted probabilities to find where we have enough TP
    tp_cumsum = np.cumsum(sorted_true)
    valid_idx = np.where(tp_cumsum >= required_tp)[0]

    if len(valid_idx) == 0:
        return 0.0, 0.0, 0.0, 0.0

    idx = valid_idx[0]
    threshold = sorted_proba[idx]

    # Compute metrics at this threshold
    y_pred_t = (y_proba >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_t == 1))
    fp = np.sum((y_true == 0) & (y_pred_t == 1))
    fn = np.sum((y_true == 1) & (y_pred_t == 0))

    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return threshold, prec, rec, f1


def plot_roc_curve(y_true, y_proba, roc_auc, save_path):
    fpr, tpr, _ = compute_roc_curve(y_true, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pr_curve(y_true, y_proba, save_path):
    precision, recall, _ = compute_pr_curve(y_true, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='#F44336', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    labels = ['Benign (0)', 'Malignant (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main(args: argparse.Namespace):
    set_seed(args.sr_no)
    os.makedirs("plots/prob1", exist_ok=True)

    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)
    print("Data Loaded")

    # Preprocess the data: median imputation
    X_train, train_medians = preprocess_data(X_train)
    X_val, _ = preprocess_data(X_val, medians=train_medians)

    # Optional feature engineering
    fe_params = {}  # store fit params for test set

    if args.winsorize:
        X_train, win_bounds = winsorize(X_train)
        X_val, _ = winsorize(X_val, bounds=win_bounds)
        fe_params['win_bounds'] = win_bounds

    if args.power_transform:
        X_train, yj_lambdas = apply_yeo_johnson(X_train)
        X_val, _ = apply_yeo_johnson(X_val, lambdas=yj_lambdas)
        fe_params['yj_lambdas'] = yj_lambdas

    if args.remove_corr:
        X_train, kept_mask = remove_correlated_features(X_train)
        X_val, _ = remove_correlated_features(X_val)
        fe_params['kept_mask'] = kept_mask

    print(f"Data Preprocessed (features: {X_train.shape[1]})")

    # Train the model
    model = GaussianNaiveBayes(
        var_smoothing=args.smoothing,
        uniform_prior=args.uniform_prior
    )
    model.fit(X_train, y_train)
    print("Model Trained")

    # Evaluate on train
    evaluate_model(model, X_train, y_train, "Train")

    # Evaluate on validation
    _, y_val_proba, val_metrics = evaluate_model(model, X_val, y_val, "Validation")

    # Threshold tuning for recall > 0.95
    threshold, prec_t, rec_t, f1_t = find_threshold_for_recall(y_val, y_val_proba, target_recall=0.95)
    print(f"\nThreshold for Recall >= 0.95:")
    print(f"Threshold:{threshold:.4f}")
    print(f"Precision:{prec_t:.4f}")
    print(f"Recall:{rec_t:.4f}")
    print(f"F1:{f1_t:.4f}")

    # Load and evaluate on test data
    test_path = os.path.join(args.data_path, "test1.csv")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        y_test = (df_test['diagnosis'] == 'M').astype(int).values
        X_test = df_test.drop('diagnosis', axis=1).values.astype(float)
        X_test, _ = preprocess_data(X_test, medians=train_medians)
        if args.winsorize:
            X_test, _ = winsorize(X_test, bounds=fe_params['win_bounds'])
        if args.power_transform:
            X_test, _ = apply_yeo_johnson(X_test, lambdas=fe_params['yj_lambdas'])
        if args.remove_corr:
            X_test, _ = remove_correlated_features(X_test)

        print("TEST SET RESULTS")

        # Default threshold (argmax)
        _, y_test_proba, test_metrics = evaluate_model(model, X_test, y_test, "Test (default threshold)")

        # With tuned threshold
        y_test_pred_tuned = (y_test_proba >= threshold).astype(int)
        cm_tuned = compute_confusion_matrix(y_test, y_test_pred_tuned)
        metrics_tuned = compute_precision_recall_f1(y_test, y_test_pred_tuned)
        acc_tuned = np.mean(y_test_pred_tuned == y_test)
        print(f"\n Test (threshold={threshold:.4f} for recall>=0.95):")
        print(f"Accuracy:    {acc_tuned:.4f}")
        print(f"Precision:   {metrics_tuned['precision_macro']:.4f}")
        print(f"Recall:      {metrics_tuned['recall_macro']:.4f}")
        print(f"F1 (macro):  {metrics_tuned['f1_macro']:.4f}")

        # Plots on test set
        plot_roc_curve(y_test, y_test_proba, test_metrics['roc_auc'],
                       "plots/prob1/roc_curve.png")
        plot_pr_curve(y_test, y_test_proba,
                      "plots/prob1/pr_curve.png")
        plot_confusion_matrix(test_metrics['confusion_matrix'],
                              "plots/prob1/confusion_matrix.png")
        plot_confusion_matrix(cm_tuned,
                              "plots/prob1/confusion_matrix_tuned.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="train1.csv")
    parser.add_argument("--smoothing", type=float, default=1e-9)
    parser.add_argument("--uniform_prior", action="store_true")
    parser.add_argument("--power_transform", action="store_true")
    parser.add_argument("--remove_corr", action="store_true")
    parser.add_argument("--winsorize", action="store_true")
    main(parser.parse_args())