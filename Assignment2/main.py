import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils import get_data, preprocess_data, set_seed

def load_and_prep_data(filepath, seed):
    """Loads dataset, splits, imputes, standardizes, and converts to PyTorch tensors."""
    X_train, y_train, X_val, y_val = get_data(filepath, seed)

    l_df = pd.read_csv(filepath)
    l_feature_names = [c for c in l_df.columns if c != 'diagnosis']

    X_train, l_train_medians = preprocess_data(X_train)
    X_val, _ = preprocess_data(X_val, f_medians=l_train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_train.shape[1], l_feature_names, scaler

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # raw logits; sigmoid applied via BCEWithLogitsLoss

class MLPTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        l_out = self.relu(self.fc1(x))
        return self.fc2(l_out)


class DeepMLPRelu(nn.Module):
    def __init__(self, f_input_dim, f_hidden_dims):
        super(DeepMLPRelu, self).__init__()
        l_layers = []
        l_prev_dim = f_input_dim
        for l_h in f_hidden_dims:
            l_layers.append(nn.Linear(l_prev_dim, l_h))
            l_layers.append(nn.ReLU())
            l_prev_dim = l_h
        l_layers.append(nn.Linear(l_prev_dim, 1))
        self.network = nn.Sequential(*l_layers)

    def forward(self, x):
        return self.network(x)

class DeepMLPSigmoid(nn.Module):
    def __init__(self, f_input_dim, f_hidden_dims):
        super(DeepMLPSigmoid, self).__init__()
        l_layers = []
        l_prev_dim = f_input_dim
        for l_h in f_hidden_dims:
            l_layers.append(nn.Linear(l_prev_dim, l_h))
            l_layers.append(nn.Sigmoid())
            l_prev_dim = l_h
        l_layers.append(nn.Linear(l_prev_dim, 1))
        self.network = nn.Sequential(*l_layers)

    def forward(self, x):
        return self.network(x)

def train_model(model, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=1000):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    l_val_losses = []
    l_train_accs = []
    l_val_accs = []

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        with torch.no_grad():
            l_train_preds = (torch.sigmoid(outputs) >= 0.5).float()
            l_train_acc = (l_train_preds == y_train).float().mean().item()
            l_train_accs.append(l_train_acc)

            model.eval()
            l_val_outputs = model(X_val)
            l_val_loss = criterion(l_val_outputs, y_val)
            l_val_losses.append(l_val_loss.item())
            l_val_preds = (torch.sigmoid(l_val_outputs) >= 0.5).float()
            l_val_acc = (l_val_preds == y_val).float().mean().item()
            l_val_accs.append(l_val_acc)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Train Acc: {l_train_acc:.4f}, Val Acc: {l_val_acc:.4f}')

    return train_losses, l_val_losses, l_train_accs, l_val_accs

def evaluate_model_f(f_model, f_X, f_y, f_dataset_name="Dataset"):
    f_model.eval()
    with torch.no_grad():
        l_logits = f_model(f_X)
        l_probs = torch.sigmoid(l_logits).numpy().ravel()
        l_preds = (l_probs >= 0.5).astype(int)
    l_y_np = f_y.numpy().ravel()

    l_metrics = {
        'accuracy': accuracy_score(l_y_np, l_preds),
        'precision': precision_score(l_y_np, l_preds),
        'recall': recall_score(l_y_np, l_preds),
        'f1': f1_score(l_y_np, l_preds),
        'roc_auc': roc_auc_score(l_y_np, l_probs),
    }
    print(f"{f_dataset_name}: acc={l_metrics['accuracy']:.4f} prec={l_metrics['precision']:.4f} "
          f"rec={l_metrics['recall']:.4f} f1={l_metrics['f1']:.4f} auc={l_metrics['roc_auc']:.4f}")
    return l_metrics, l_probs, l_preds


def plot_loss_vs_iterations_f(f_loss_dict, f_save_path):
    plt.figure(figsize=(10, 6))
    for l_lr, l_losses in f_loss_dict.items():
        plt.plot(l_losses, label=f'alpha = {l_lr}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.title('Loss vs. Iterations (Logistic Regression)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f_save_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_feature_importance_f(f_model, f_feature_names, f_save_path):
    l_weights = f_model.linear.weight.data.numpy().ravel()
    l_bias = f_model.linear.bias.data.numpy().ravel()

    l_abs_weights = np.abs(l_weights)
    l_sorted_idx = np.argsort(l_abs_weights)[::-1].copy()

    print(f"Bias: {l_bias[0]:.4f}")
    for l_i, l_idx in enumerate(l_sorted_idx[:10]):
        print(f"{l_i+1:>2}. {f_feature_names[l_idx]:<28} w={l_weights[l_idx]:>8.4f}  |w|={l_abs_weights[l_idx]:.4f}")

    l_top_k = 10
    l_top_idx = l_sorted_idx[:l_top_k]
    plt.figure(figsize=(10, 6))
    l_colors = ['#E53935' if l_weights[i] > 0 else '#1E88E5' for i in l_top_idx]
    plt.barh(range(l_top_k), l_abs_weights[l_top_idx][::-1], color=l_colors[::-1])
    plt.yticks(range(l_top_k), [f_feature_names[i] for i in l_top_idx][::-1], fontsize=10)
    plt.xlabel('Weight', fontsize=12)
    plt.title('Top 10 Feature Importances (Logistic Regression)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    l_top3_idx = l_sorted_idx[:3]
    l_top3_names = [f_feature_names[i] for i in l_top3_idx]
    return l_weights, l_top3_idx, l_top3_names


def plot_decision_boundary_f(f_model_class, f_X_train, f_y_train, f_X_val, f_y_val,
                             f_top2_idx, f_top2_names, f_save_path, f_lr=0.01, f_epochs=1000):
    l_X_train_2d = f_X_train[:, f_top2_idx]
    l_X_val_2d = f_X_val[:, f_top2_idx]

    l_model_2d = f_model_class(2)
    l_criterion = nn.BCEWithLogitsLoss()
    l_optimizer = optim.SGD(l_model_2d.parameters(), lr=f_lr)

    for l_epoch in range(f_epochs):
        l_model_2d.train()
        l_out = l_model_2d(l_X_train_2d)
        l_loss = l_criterion(l_out, f_y_train)
        l_optimizer.zero_grad()
        l_loss.backward()
        l_optimizer.step()

    l_model_2d.eval()
    l_w = l_model_2d.linear.weight.data.numpy().ravel()
    l_b = l_model_2d.linear.bias.data.numpy().item()

    l_X_all = torch.cat([l_X_train_2d, l_X_val_2d], dim=0).numpy()
    l_y_all = torch.cat([f_y_train, f_y_val], dim=0).numpy().ravel()

    l_x_min, l_x_max = l_X_all[:, 0].min() - 0.5, l_X_all[:, 0].max() + 0.5
    l_y_min, l_y_max = l_X_all[:, 1].min() - 0.5, l_X_all[:, 1].max() + 0.5

    l_xx, l_yy = np.meshgrid(np.linspace(l_x_min, l_x_max, 300),
                              np.linspace(l_y_min, l_y_max, 300))
    l_grid = torch.tensor(np.c_[l_xx.ravel(), l_yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        l_zz = torch.sigmoid(l_model_2d(l_grid)).numpy().reshape(l_xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(l_xx, l_yy, l_zz, levels=50, cmap='RdBu_r', alpha=0.6)
    plt.contour(l_xx, l_yy, l_zz, levels=[0.5], colors='black', linewidths=2)

    l_colors_map = {0: '#1E88E5', 1: '#E53935'}
    l_labels_map = {0: 'Benign', 1: 'Malignant'}
    for l_cls in [0, 1]:
        l_mask = l_y_all == l_cls
        plt.scatter(l_X_all[l_mask, 0], l_X_all[l_mask, 1],
                    c=l_colors_map[l_cls], label=l_labels_map[l_cls],
                    edgecolors='k', alpha=0.7, s=40)
    plt.xlabel(f_top2_names[0], fontsize=12)
    plt.ylabel(f_top2_names[1], fontsize=12)
    plt.title('Decision Boundary (Top 2 Features)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.colorbar(label='P(Malignant)')
    plt.tight_layout()
    plt.savefig(f_save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_train_val_accuracy_f(f_train_accs, f_val_accs, f_title, f_save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(f_train_accs, label='Train Accuracy', color='#1E88E5')
    plt.plot(f_val_accs, label='Validation Accuracy', color='#E53935')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f_title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f_save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_comparison_f(f_losses_dict, f_title, f_save_path):
    plt.figure(figsize=(10, 6))
    for l_label, l_losses in f_losses_dict.items():
        plt.plot(l_losses, label=l_label)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.title(f_title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f_save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_eda_plots_f(f_filepath, f_plots_dir):
    l_df = pd.read_csv(f_filepath)

    l_counts = l_df['diagnosis'].value_counts()
    plt.figure(figsize=(6, 5))
    l_colors = ['#1E88E5', '#E53935']
    l_counts.plot(kind='bar', color=l_colors, edgecolor='black')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks([0, 1], ['Benign (B)', 'Malignant (M)'], rotation=0)
    for l_i, l_v in enumerate(l_counts):
        plt.text(l_i, l_v + 5, str(l_v), ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(f_plots_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    l_features = l_df.drop('diagnosis', axis=1)
    l_missing = l_features.isnull().sum()
    l_missing_cols = l_missing[l_missing > 0]
    if len(l_missing_cols) > 0:
        plt.figure(figsize=(10, 5))
        l_missing_cols.plot(kind='bar', color='#FF7043', edgecolor='black')
        plt.title('Missing Values per Feature', fontsize=14, fontweight='bold')
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Missing Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(f_plots_dir, 'missing_values.png'), dpi=150, bbox_inches='tight')
        plt.close()

    l_numeric = l_features.apply(pd.to_numeric, errors='coerce')
    l_corr = l_numeric.corr()
    plt.figure(figsize=(14, 12))
    l_im = plt.imshow(l_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(l_im, shrink=0.8)
    plt.xticks(range(len(l_corr.columns)), l_corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(l_corr.columns)), l_corr.columns, fontsize=7)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(f_plots_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    l_seed = 34
    set_seed(l_seed)
    l_plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    for l_sub in ['eda', 'part1', 'part2', 'comparison']:
        os.makedirs(os.path.join(l_plots_dir, l_sub), exist_ok=True)

    generate_eda_plots_f('data.csv', os.path.join(l_plots_dir, 'eda'))

    X_train, y_train, X_val, y_val, input_dim, l_feature_names, l_scaler = \
        load_and_prep_data('data.csv', l_seed)
    print(f"Input dim: {input_dim}, Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    print("1: LOGISTIC REGRESSION")
    l_learning_rates = [0.1, 0.01, 0.001]
    l_lr_results = {}
    l_lr_loss_dict = {}

    for l_lr in l_learning_rates:
        print(f"Training LR with alpha={l_lr}")
        set_seed(l_seed)
        l_model_lr = LogisticRegressionTorch(input_dim)
        l_train_losses, l_val_losses, l_train_accs, l_val_accs = \
            train_model(l_model_lr, X_train, y_train, X_val, y_val,
                        learning_rate=l_lr, epochs=1000)
        l_lr_results[l_lr] = {
            'model': l_model_lr,
            'train_losses': l_train_losses,
            'val_losses': l_val_losses,
            'train_accs': l_train_accs,
            'val_accs': l_val_accs,
        }
        l_lr_loss_dict[l_lr] = l_train_losses

    plot_loss_vs_iterations_f(l_lr_loss_dict,
                              os.path.join(l_plots_dir, 'part1', 'lr_loss_vs_iterations.png'))

    l_best_lr = min(l_lr_results, key=lambda k: l_lr_results[k]['val_losses'][-1])
    l_best_lr_model = l_lr_results[l_best_lr]['model']
    print(f"Best LR: alpha={l_best_lr}, val_loss={l_lr_results[l_best_lr]['val_losses'][-1]:.4f}")
    l_lr_metrics, l_lr_probs, l_lr_preds = evaluate_model_f(
        l_best_lr_model, X_val, y_val, "Logistic Regression (Test)")

    l_weights, l_top3_idx, l_top3_names = extract_feature_importance_f(
        l_best_lr_model, l_feature_names,
        os.path.join(l_plots_dir, 'part1', 'lr_feature_importance.png'))
    print(f"Top 3 features: {l_top3_names}")

    l_top2_idx = l_top3_idx[:2]
    l_top2_names = l_top3_names[:2]
    plot_decision_boundary_f(
        LogisticRegressionTorch, X_train, y_train, X_val, y_val,
        l_top2_idx, l_top2_names,
        os.path.join(l_plots_dir, 'part1', 'lr_decision_boundary.png'),
        f_lr=l_best_lr, f_epochs=1000)

    print("PART 2: MLP")

    l_mlp_lr = 0.01
    l_mlp_epochs = 1000
    l_mlp_results = {}

    l_mlp_configs = [
        ('Linear Baseline (0 hidden)', lambda: LogisticRegressionTorch(input_dim)),
        ('Wide Network (1x100)',       lambda: MLPTorch(input_dim, 100)),
        ('Deep Network ReLU (3x10)',   lambda: DeepMLPRelu(input_dim, [10, 10, 10])),
    ]

    for l_name, l_make_model in l_mlp_configs:
        print(f"Training: {l_name}")
        set_seed(l_seed)
        l_model_mlp = l_make_model()
        l_train_losses, l_val_losses, l_train_accs, l_val_accs = \
            train_model(l_model_mlp, X_train, y_train, X_val, y_val,
                        learning_rate=l_mlp_lr, epochs=l_mlp_epochs)
        l_mlp_results[l_name] = {
            'model': l_model_mlp,
            'train_losses': l_train_losses,
            'val_losses': l_val_losses,
            'train_accs': l_train_accs,
            'val_accs': l_val_accs,
        }

    print("Sigmoid vs ReLU (Deep 3x10)")
    l_activation_losses = {}

    print(f"Training deep 3x10 with RELU")
    set_seed(l_seed)
    l_model_relu = DeepMLPRelu(input_dim, [10, 10, 10])
    l_relu_tl, l_relu_vl, l_relu_ta, l_relu_va = \
        train_model(l_model_relu, X_train, y_train, X_val, y_val,
                    learning_rate=l_mlp_lr, epochs=l_mlp_epochs)
    l_activation_losses['Deep 3x10 (relu)'] = l_relu_tl

    print(f"Training deep 3x10 with sigmoid")
    set_seed(l_seed)
    l_model_sigmoid = DeepMLPSigmoid(input_dim, [10, 10, 10])
    l_sig_tl, l_sig_vl, l_sig_ta, l_sig_va = \
        train_model(l_model_sigmoid, X_train, y_train, X_val, y_val,
                    learning_rate=l_mlp_lr, epochs=l_mlp_epochs)
    l_activation_losses['Deep 3x10 (sigmoid)'] = l_sig_tl

    l_mlp_results['Deep Network Sigmoid (3x10)'] = {
        'model': l_model_sigmoid,
        'train_losses': l_sig_tl,
        'val_losses': l_sig_vl,
        'train_accs': l_sig_ta,
        'val_accs': l_sig_va,
    }

    plot_convergence_comparison_f(
        l_activation_losses,
        'Convergence: Sigmoid vs ReLU (Deep Network 3x10)',
        os.path.join(l_plots_dir, 'part2', 'mlp_sigmoid_vs_relu.png'))

    for l_name, l_res in l_mlp_results.items():
        l_safe_name = l_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plot_train_val_accuracy_f(
            l_res['train_accs'], l_res['val_accs'],
            f'Train vs Val Accuracy: {l_name}',
            os.path.join(l_plots_dir, 'part2', f'mlp_train_val_acc_{l_safe_name}.png'))

    l_all_metrics = {'Logistic Regression': l_lr_metrics}
    for l_name, l_res in l_mlp_results.items():
        l_metrics, _, _ = evaluate_model_f(l_res['model'], X_val, y_val, l_name)
        l_all_metrics[l_name] = l_metrics

    print("Comparison")
    l_header = f"{'Model':<35}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'F1':<10}{'AUC':<10}"
    print(l_header)
    print('-' * len(l_header))
    for l_name, l_m in l_all_metrics.items():
        print(f"{l_name:<35}{l_m['accuracy']:<10.4f}{l_m['precision']:<10.4f}"
              f"{l_m['recall']:<10.4f}{l_m['f1']:<10.4f}{l_m['roc_auc']:<10.4f}")

    l_model_names = list(l_all_metrics.keys())
    l_metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    l_x_pos = np.arange(len(l_model_names))
    l_width = 0.15

    plt.figure(figsize=(14, 7))
    for l_i, l_metric in enumerate(l_metric_names):
        l_vals = [l_all_metrics[m][l_metric] for m in l_model_names]
        plt.bar(l_x_pos + l_i * l_width, l_vals, l_width, label=l_metric.capitalize())
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Comparison: All Metrics', fontsize=14, fontweight='bold')
    plt.xticks(l_x_pos + l_width * 2, l_model_names, rotation=30, ha='right', fontsize=9)
    plt.ylim(0.7, 1.05)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(l_plots_dir, 'comparison', 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Done. Plots in {l_plots_dir}")
