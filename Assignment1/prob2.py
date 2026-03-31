import os
import time
import argparse
import numpy as np
from utils import *
from model import *

def main(args: argparse.Namespace):
    assert args.run_id is not None and 0 < args.run_id < 6, "Invalid run_id"
    set_seed(args.sr_no+args.run_id)

    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)

    # Preprocess: median imputation fit on training split, apply to validation split
    X_train, train_medians = preprocess_data(X_train)
    X_val, _ = preprocess_data(X_val, medians=train_medians)

    # Optional feature engineering (kept consistent with prob1)
    fe_params = {}
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

    print("Data Loaded and Preprocessed")

    # Train the model (incremental update approach)
    model = GaussianNaiveBayes(var_smoothing=args.smoothing)
    accs = []
    times_incremental = []
    times_retrain = []
    total_items = 20
    batch_size = 10

    all_train_idxs = np.arange(X_train.shape[0])
    # Random initial labeled pool to ensure both classes appear early (reproducible due to set_seed)
    idxs = np.random.choice(all_train_idxs, size=min(total_items, len(all_train_idxs)), replace=False)
    remaining_idxs = np.setdiff1d(all_train_idxs, idxs)
    # Pre-shuffle remaining pool for deterministic passive sampling
    remaining_idxs = np.random.permutation(remaining_idxs)

    for i in range(1, 40):
        if len(idxs) > X_train.shape[0]:
            break

        n_timing_reps = 200

        if i == 1:
            # Initial fit on the initial labeled pool
            model.fit(X_train[idxs], y_train[idxs])

            # Timing: measure over many reps for stability
            t0 = time.perf_counter()
            for _ in range(n_timing_reps):
                m_tmp = GaussianNaiveBayes(var_smoothing=args.smoothing)
                m_tmp.fit(X_train[idxs], y_train[idxs])
            t_ret = (time.perf_counter() - t0) / n_timing_reps
            t_inc = t_ret  # same operation for first iteration
        else:
            # --- Incremental update: only new samples ---
            model.fit(X_train[new_idxs], y_train[new_idxs], update=True)

            # Timing: incremental update only (constant batch size)
            # Pre-build a model with the old data, then time only the update step
            m_tmp = GaussianNaiveBayes(var_smoothing=args.smoothing)
            old_idxs = idxs[:len(idxs)-len(new_idxs)]
            m_tmp.fit(X_train[old_idxs], y_train[old_idxs])
            t0 = time.perf_counter()
            for _ in range(n_timing_reps):
                m_tmp.fit(X_train[new_idxs], y_train[new_idxs], update=True)
            t_inc = (time.perf_counter() - t0) / n_timing_reps

            # Timing: retrain from scratch (repeated)
            t0 = time.perf_counter()
            for _ in range(n_timing_reps):
                m_tmp = GaussianNaiveBayes(var_smoothing=args.smoothing)
                m_tmp.fit(X_train[idxs], y_train[idxs])
            t_ret = (time.perf_counter() - t0) / n_timing_reps

        times_incremental.append(t_inc)
        times_retrain.append(t_ret)

        y_preds = model.predict(X_val)
        val_acc = np.mean(y_preds == y_val)
        print(f"{total_items} items - Val acc: {val_acc:.4f}  (inc: {t_inc*1000:.2f}ms  retrain: {t_ret*1000:.2f}ms)")
        accs.append(val_acc)

        if len(remaining_idxs) == 0:
            break

        current_batch_size = min(batch_size, len(remaining_idxs))

        if args.is_active:
            # Uncertainty sampling: pick samples with probability closest to 0.5
            proba_pool = model.predict_proba(X_train[remaining_idxs])[:, 1]
            uncertainty = np.abs(proba_pool - 0.5)
            pick_order = np.argsort(uncertainty)
            new_idxs = remaining_idxs[pick_order[:current_batch_size]]
        else:
            # Passive sampling: take next chunk (deterministic given shuffled data)
            new_idxs = remaining_idxs[:current_batch_size]

        idxs = np.concatenate([idxs, new_idxs])
        mask = ~np.isin(remaining_idxs, new_idxs)
        remaining_idxs = remaining_idxs[mask]

        total_items += current_batch_size

    accs = np.array(accs)
    times_incremental = np.array(times_incremental)
    times_retrain = np.array(times_retrain)
    os.makedirs(args.logs_path, exist_ok=True)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}.npy", accs)
    np.save(f"{args.logs_path}/time_inc_{args.run_id}_{args.is_active}.npy", times_incremental)
    np.save(f"{args.logs_path}/time_retrain_{args.run_id}_{args.is_active}.npy", times_retrain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--is_active", action="store_true")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="train1.csv")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--smoothing", type=float, default=1e-9)
    parser.add_argument("--power_transform", action="store_true")
    parser.add_argument("--remove_corr", action="store_true")
    parser.add_argument("--winsorize", action="store_true")
    main(parser.parse_args())