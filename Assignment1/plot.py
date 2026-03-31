import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args: argparse.Namespace):
    assert os.path.exists(args.logs_path), "Invalid logs path"
    for i in [True, False]:
        for j in range(1, 6):
            assert os.path.exists(os.path.join(args.logs_path, f"run_{j}_{i}.npy")),\
                f"File run_{j}_{i}.npy not found in {args.logs_path}"

    # Load accuracy curves for passive (False) and active (True)
    passive_runs = []
    active_runs = []
    for j in range(1, 6):
        passive_runs.append(np.load(os.path.join(args.logs_path, f"run_{j}_False.npy")))
        active_runs.append(np.load(os.path.join(args.logs_path, f"run_{j}_True.npy")))

    # Ensure all runs have the same length
    min_len = min(min(len(r) for r in passive_runs), min(len(r) for r in active_runs))
    passive_runs = np.array([r[:min_len] for r in passive_runs])
    active_runs = np.array([r[:min_len] for r in active_runs])

    passive_mean = passive_runs.mean(axis=0)
    passive_std = passive_runs.std(axis=0)
    active_mean = active_runs.mean(axis=0)
    active_std = active_runs.std(axis=0)

    # X-axis: number of labeled samples (starts at 20, increments of 10)
    n_points = min_len
    x = np.array([20 + 10 * i for i in range(n_points)])

    os.makedirs("plots/prob2", exist_ok=True)

    # --- Plot 1: Learning curves (active vs passive) ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, passive_mean, color='#2196F3', lw=2, label='Random (passive)')
    plt.fill_between(x, passive_mean - passive_std, passive_mean + passive_std,
                     color='#2196F3', alpha=0.2)
    plt.plot(x, active_mean, color='#F44336', lw=2, label='ALS (active)')
    plt.fill_between(x, active_mean - active_std, active_mean + active_std,
                     color='#F44336', alpha=0.2)
    plt.axhline(y=args.supervised_accuracy, color='green', linestyle='--', lw=2,
                label=f'Supervised baseline ({args.supervised_accuracy:.4f})')
    plt.xlabel('Number of Labeled Samples', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Active Learning vs Random Sampling', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/prob2/learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/prob2/learning_curves.png")

    # --- Plot 2: Training time comparison (retrain vs incremental) ---
    # Load timing logs (use passive run_id=1 as representative)
    has_timing = True
    try:
        time_inc = np.load(os.path.join(args.logs_path, "time_inc_1_False.npy"))
        time_ret = np.load(os.path.join(args.logs_path, "time_retrain_1_False.npy"))
    except FileNotFoundError:
        has_timing = False
        print("Warning: Timing logs not found, skipping timing plot.")

    if has_timing:
        t_len = min(len(time_inc), len(time_ret))
        time_inc = time_inc[:t_len]
        time_ret = time_ret[:t_len]
        x_time = np.array([20 + 10 * i for i in range(t_len)])

        plt.figure(figsize=(10, 6))
        plt.plot(x_time, time_ret * 1000, color='#FF9800', lw=2, marker='o', markersize=3,
                 label='Retrain from scratch')
        plt.plot(x_time, time_inc * 1000, color='#4CAF50', lw=2, marker='s', markersize=3,
                 label='Incremental update')
        plt.xlabel('Number of Labeled Samples', fontsize=12)
        plt.ylabel('Training Time (ms)', fontsize=12)
        plt.title('Training Time: Retrain vs Incremental Update', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/prob2/timing_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: plots/prob2/timing_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--supervised_accuracy", type=float, required=True)
    main(parser.parse_args())