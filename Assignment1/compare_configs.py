"""
Compare all feature engineering variants x smoothing x prior combinations.
Finds the best configuration on validation set for Problem 1.
"""
import os
import argparse
import numpy as np
import pandas as pd
from utils import *
from model import *

def run_config(X_train, y_train, X_val, y_val, smoothing, uniform_prior, variant_name):
    """Train and evaluate a single configuration. Returns val accuracy and metrics."""
    model = GaussianNaiveBayes(var_smoothing=smoothing, uniform_prior=uniform_prior)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    val_acc = np.mean(y_val_pred == y_val)
    metrics = compute_precision_recall_f1(y_val, y_val_pred)
    roc_auc = compute_roc_auc(y_val, y_val_proba)

    return {
        'variant': variant_name,
        'smoothing': smoothing,
        'prior': 'uniform' if uniform_prior else 'empirical',
        'val_acc': val_acc,
        'val_f1_macro': metrics['f1_macro'],
        'val_f1_weighted': metrics['f1_weighted'],
        'val_roc_auc': roc_auc,
        'val_precision_macro': metrics['precision_macro'],
        'val_recall_macro': metrics['recall_macro'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="data")
    args = parser.parse_args()

    set_seed(args.sr_no)

    # Load data
    X_train_raw, y_train, X_val_raw, y_val = get_data(
        path=os.path.join(args.data_path, "train1.csv"), seed=args.sr_no)

    # Preprocessing: median imputation (always applied first)
    X_train_imp, medians = preprocess_data(X_train_raw)
    X_val_imp, _ = preprocess_data(X_val_raw, medians=medians)

    # Build variants
    variants = {}

    # A: Vanilla (imputation only)
    variants['A_vanilla'] = (X_train_imp, X_val_imp)

    # B: Power Transform (Yeo-Johnson)
    X_train_yj, yj_lambdas = apply_yeo_johnson(X_train_imp)
    X_val_yj, _ = apply_yeo_johnson(X_val_imp, lambdas=yj_lambdas)
    variants['B_power_transform'] = (X_train_yj, X_val_yj)

    # C: Remove Correlated Features
    X_train_rc, kept_mask = remove_correlated_features(X_train_imp)
    X_val_rc, _ = remove_correlated_features(X_val_imp)
    variants['C_remove_corr'] = (X_train_rc, X_val_rc)

    # D: Power Transform + Remove Correlated
    X_train_yj_rc, _ = remove_correlated_features(X_train_yj)
    X_val_yj_rc, _ = remove_correlated_features(X_val_yj)
    variants['D_power+remove'] = (X_train_yj_rc, X_val_yj_rc)

    # E: Winsorize + Vanilla
    X_train_win, win_bounds = winsorize(X_train_imp)
    X_val_win, _ = winsorize(X_val_imp, bounds=win_bounds)
    variants['E_winsorize'] = (X_train_win, X_val_win)

    # F: Winsorize + Power Transform
    X_train_win_yj, yj_lam2 = apply_yeo_johnson(X_train_win)
    X_val_win_yj, _ = apply_yeo_johnson(X_val_win, lambdas=yj_lam2)
    variants['F_winsor+power'] = (X_train_win_yj, X_val_win_yj)

    # G: Winsorize + Power Transform + Remove Correlated
    X_train_win_yj_rc, _ = remove_correlated_features(X_train_win_yj)
    X_val_win_yj_rc, _ = remove_correlated_features(X_val_win_yj)
    variants['G_winsor+power+remove'] = (X_train_win_yj_rc, X_val_win_yj_rc)

    smoothings = [1e-9, 1e-6, 1e-3, 1e-1]
    priors = [False, True]  # empirical, uniform

    results = []
    total = len(variants) * len(smoothings) * len(priors)
    print(f"Running {total} configurations...\n")

    for vname, (X_tr, X_va) in variants.items():
        for sm in smoothings:
            for up in priors:
                res = run_config(X_tr, y_train, X_va, y_val, sm, up, vname)
                results.append(res)

    df = pd.DataFrame(results)
    df = df.sort_values('val_acc', ascending=False).reset_index(drop=True)

    # Printing full table
    print("\n" + "ALL RESULTS (sorted by val_acc)")
    print(df.to_string(index=False, float_format='%.4f'))

    # Printing top 10
    print("\n" + "TOP 10 CONFIGURATIONS")
    print(df.head(10).to_string(index=False, float_format='%.4f'))

    # Best overall
    best = df.iloc[0]
    print(f"\n" + "BEST CONFIG: {best['variant']}")
    print(f"Smoothing: {best['smoothing']}")
    print(f"Prior: {best['prior']}")
    print(f"Val Accuracy: {best['val_acc']:.4f}")
    print(f"Val F1 (macro): {best['val_f1_macro']:.4f}")
    print(f"Val ROC-AUC: {best['val_roc_auc']:.4f}")

    # Save results
    os.makedirs("plots/prob1", exist_ok=True)
    df.to_csv("plots/prob1/config_comparison.csv", index=False)
    print("\nSaved: plots/prob1/config_comparison.csv")

    # Summary per variant (best smoothing/prior)
    print("\n" + "BEST PER VARIANT (best smoothing+prior for each)")
    summary = df.loc[df.groupby('variant')['val_acc'].idxmax()]
    summary = summary.sort_values('val_acc', ascending=False)
    print(summary[['variant', 'smoothing', 'prior', 'val_acc', 'val_f1_macro', 'val_roc_auc']].to_string(index=False, float_format='%.4f'))


if __name__ == "__main__":
    main()
