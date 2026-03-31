"""
GNB Assumption Analysis for Breast Cancer Wisconsin Dataset
Generates plots and statistics for the report on assumption violations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.1)
SAVE_DIR = os.path.join("plots", "assumption_analysis")
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(r'data\train1.csv')
features = [c for c in df.columns if c != 'diagnosis']
classes = ['B', 'M']
class_colors = {'B': '#2196F3', 'M': '#F44336'}
class_labels = {'B': 'Benign', 'M': 'Malignant'}

# =============================================================================
# 1. NORMALITY ANALYSIS — Shapiro-Wilk + D'Agostino-Pearson tests
# =============================================================================
print("=" * 60)
print("1. NORMALITY ANALYSIS")
print("=" * 60)

normality_results = []
for cls in classes:
    subset = df[df['diagnosis'] == cls]
    for feat in features:
        vals = subset[feat].dropna().values
        if len(vals) >= 20:
            sw_stat, sw_p = stats.shapiro(vals)
            dag_stat, dag_p = stats.normaltest(vals)
            skew = stats.skew(vals)
            kurt = stats.kurtosis(vals)
            normality_results.append({
                'class': cls, 'feature': feat,
                'shapiro_stat': sw_stat, 'shapiro_p': sw_p,
                'dagostino_stat': dag_stat, 'dagostino_p': dag_p,
                'skewness': skew, 'kurtosis': kurt,
                'n': len(vals)
            })

norm_df = pd.DataFrame(normality_results)
norm_df['shapiro_pass'] = norm_df['shapiro_p'] >= 0.05
norm_df['dagostino_pass'] = norm_df['dagostino_p'] >= 0.05

# Summary table
print("\nShapiro-Wilk Test Summary (alpha=0.05):")
for cls in classes:
    sub = norm_df[norm_df['class'] == cls]
    n_pass = sub['shapiro_pass'].sum()
    print(f"  Class {cls} ({class_labels[cls]}): {n_pass}/{len(sub)} pass ({n_pass/len(sub)*100:.1f}%)")

total_pass = norm_df['shapiro_pass'].sum()
print(f"  Overall: {total_pass}/{len(norm_df)} pass ({total_pass/len(norm_df)*100:.1f}%)")

print("\nD'Agostino-Pearson Test Summary (alpha=0.05):")
for cls in classes:
    sub = norm_df[norm_df['class'] == cls]
    n_pass = sub['dagostino_pass'].sum()
    print(f"  Class {cls} ({class_labels[cls]}): {n_pass}/{len(sub)} pass ({n_pass/len(sub)*100:.1f}%)")

# --- PLOT 1a: Normality test p-value heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for idx, cls in enumerate(classes):
    sub = norm_df[norm_df['class'] == cls].set_index('feature')['shapiro_p']
    pvals = sub.values.reshape(-1, 1)
    log_pvals = -np.log10(np.clip(pvals, 1e-30, 1))

    ax = axes[idx]
    im = ax.imshow(log_pvals, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=15)
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['-log10(p)'])
    ax.set_title(f'Class: {class_labels[cls]}', fontsize=14, fontweight='bold')
    ax.axhline(y=-0.5, color='black', linewidth=0.5)

    # Mark pass/fail
    for i, (feat, p) in enumerate(sub.items()):
        marker = 'Y' if p >= 0.05 else 'N'
        ax.text(0, i, f'[{marker}] p={p:.2e}', ha='center', va='center', fontsize=7,
                fontweight='bold', color='white' if -np.log10(max(p, 1e-30)) > 7 else 'black')

plt.colorbar(im, ax=axes, label='-log10(p-value)', shrink=0.8)
fig.suptitle('Shapiro-Wilk Normality Test Results\n([Y] = normal at alpha=0.05, [N] = non-normal)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '1a_normality_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1a_normality_heatmap.png")

# --- PLOT 1b: Histogram + Gaussian overlay for worst and best features ---
# Pick 4 worst (most non-normal) and 2 best (most normal) per class
worst_feats = norm_df.nsmallest(6, 'shapiro_p')[['class', 'feature']].values
best_feats = norm_df.nlargest(4, 'shapiro_p')[['class', 'feature']].values
selected = np.vstack([worst_feats, best_feats])

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()
for i, (cls, feat) in enumerate(selected):
    ax = axes[i]
    vals = df[df['diagnosis'] == cls][feat].dropna().values
    mu, sigma = vals.mean(), vals.std()

    ax.hist(vals, bins=25, density=True, alpha=0.6, color=class_colors[cls],
            edgecolor='white', linewidth=0.5)
    x_range = np.linspace(vals.min() - sigma, vals.max() + sigma, 200)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'k-', lw=2, label='Gaussian fit')

    sw_p = norm_df[(norm_df['class'] == cls) & (norm_df['feature'] == feat)]['shapiro_p'].values[0]
    status = "NORMAL" if sw_p >= 0.05 else "NON-NORMAL"
    ax.set_title(f'{feat}\nClass={cls}, p={sw_p:.2e} ({status})', fontsize=8, fontweight='bold')
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(len(selected), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Feature Distributions vs Gaussian Fit\n(Top: 6 most non-normal, Bottom-right: 4 most normal)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '1b_distribution_examples.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1b_distribution_examples.png")

# --- PLOT 1c: Q-Q plots for selected features ---
# Pick 3 worst non-normal and 3 best normal
worst3 = norm_df.nsmallest(3, 'shapiro_p')[['class', 'feature']].values
best3 = norm_df.nlargest(3, 'shapiro_p')[['class', 'feature']].values
qq_selected = np.vstack([worst3, best3])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (cls, feat) in enumerate(qq_selected):
    ax = axes[i]
    vals = df[df['diagnosis'] == cls][feat].dropna().values
    (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
    ax.scatter(osm, osr, s=10, alpha=0.6, color=class_colors[cls], edgecolors='none')
    ax.plot(osm, slope * np.array(osm) + intercept, 'k--', lw=1.5)

    sw_p = norm_df[(norm_df['class'] == cls) & (norm_df['feature'] == feat)]['shapiro_p'].values[0]
    status = "NORMAL" if sw_p >= 0.05 else "NON-NORMAL"
    ax.set_title(f'{feat}\nClass={cls}, R²={r**2:.4f}, p={sw_p:.2e}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=8)
    ax.set_ylabel('Sample Quantiles', fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle('Q-Q Plots: Top row = most non-normal, Bottom row = most normal',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '1c_qq_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1c_qq_plots.png")

# =============================================================================
# 2. FEATURE INDEPENDENCE (CORRELATION) ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("2. FEATURE INDEPENDENCE ANALYSIS")
print("=" * 60)

# --- PLOT 2a: Correlation heatmaps per class ---
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
for idx, cls in enumerate(classes):
    subset = df[df['diagnosis'] == cls][features].dropna()
    corr = subset.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    ax = axes[idx]
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                xticklabels=[f.replace('_', '\n') for f in features],
                yticklabels=[f.replace('_', '\n') for f in features],
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    ax.set_title(f'Class: {class_labels[cls]}', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=5.5, rotation=45)

fig.suptitle('Feature Correlation Matrices by Class\n(Conditional Independence Assumption Check)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '2a_correlation_heatmaps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 2a_correlation_heatmaps.png")

# --- PLOT 2b: Distribution of pairwise correlations ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
thresholds = [0.5, 0.7, 0.9]
for idx, cls in enumerate(classes):
    subset = df[df['diagnosis'] == cls][features].dropna()
    corr = subset.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_vals = upper.stack().values

    ax = axes[idx]
    ax.hist(corr_vals, bins=50, alpha=0.7, color=class_colors[cls], edgecolor='white')
    for t in thresholds:
        count_above = (corr_vals > t).sum()
        ax.axvline(x=t, color='black', linestyle='--', alpha=0.5)
        ax.text(t + 0.01, ax.get_ylim()[1] * 0.9, f'>{t}: {count_above}',
                fontsize=8, fontweight='bold')
    ax.set_xlabel('|Pearson r|', fontsize=11)
    ax.set_ylabel('Count of feature pairs', fontsize=11)
    ax.set_title(f'Class: {class_labels[cls]}\nMean |r|={corr_vals.mean():.3f}, Median={np.median(corr_vals):.3f}',
                 fontsize=12, fontweight='bold')

fig.suptitle('Distribution of Pairwise Feature Correlations (|r|)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '2b_correlation_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 2b_correlation_distribution.png")

# Print highly correlated pairs
print("\nHighly correlated feature pairs (|r| > 0.9):")
for cls in classes:
    subset = df[df['diagnosis'] == cls][features].dropna()
    corr = subset.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    for i in range(len(upper.index)):
        for j in range(len(upper.columns)):
            if upper.iloc[i, j] > 0.9:
                pairs.append((upper.index[i], upper.columns[j], upper.iloc[i, j]))
    pairs.sort(key=lambda x: -x[2])
    print(f"\n  Class {cls} ({class_labels[cls]}): {len(pairs)} pairs")
    for f1, f2, c in pairs:
        print(f"    {f1} <-> {f2}: r={c:.4f}")

# =============================================================================
# 3. SKEWNESS & KURTOSIS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("3. SKEWNESS & KURTOSIS ANALYSIS")
print("=" * 60)

# --- PLOT 3: Skewness and Kurtosis bar chart ---
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for idx, cls in enumerate(classes):
    sub = norm_df[norm_df['class'] == cls].set_index('feature')
    x = np.arange(len(features))
    width = 0.35

    ax = axes[idx]
    bars1 = ax.bar(x - width / 2, sub.loc[features, 'skewness'], width,
                   label='Skewness', color=class_colors[cls], alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width / 2, sub.loc[features, 'kurtosis'], width,
                   label='Excess Kurtosis', color=class_colors[cls], alpha=0.4, edgecolor='white',
                   hatch='//')

    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='|Skew|=1 threshold')
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_mean', '_m').replace('_worst', '_w').replace('_se', '_s')
                        for f in features], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Class: {class_labels[cls]}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    n_skewed = (sub['skewness'].abs() > 1).sum()
    ax.text(0.98, 0.95, f'{n_skewed}/{len(features)} features with |skew|>1',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('Skewness & Excess Kurtosis per Feature by Class\n(Gaussian: skew=0, kurtosis=0)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '3_skewness_kurtosis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 3_skewness_kurtosis.png")

# =============================================================================
# 4. MISSING DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("4. MISSING DATA ANALYSIS")
print("=" * 60)

missing = df[features].isnull().sum()
missing_pct = missing / len(df) * 100

print(f"Total samples: {len(df)}")
print(f"Features with missing values: {(missing > 0).sum()}/{len(features)}")
print(f"Total missing values: {missing.sum()}")
print(f"Max missing in a feature: {missing.max()} ({missing_pct.max():.2f}%)")

# --- PLOT 4: Missing data bar chart ---
fig, ax = plt.subplots(figsize=(14, 5))
missing_feats = missing[missing > 0].sort_values(ascending=False)
bars = ax.bar(range(len(missing_feats)), missing_feats.values, color='#FF9800', edgecolor='white')
ax.set_xticks(range(len(missing_feats)))
ax.set_xticklabels(missing_feats.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Number of Missing Values', fontsize=11)
ax.set_title(f'Missing Values per Feature\n({(missing > 0).sum()}/{len(features)} features affected, '
             f'{missing.sum()} total missing out of {len(df)*len(features)} values)',
             fontsize=13, fontweight='bold')

for bar, val in zip(bars, missing_feats.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0, missing_feats.max() + 1)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '4_missing_data.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 4_missing_data.png")

# =============================================================================
# 5. CLASS IMBALANCE
# =============================================================================
print("\n" + "=" * 60)
print("5. CLASS DISTRIBUTION")
print("=" * 60)

vc = df['diagnosis'].value_counts()
print(f"Benign (B): {vc['B']} ({vc['B']/len(df)*100:.1f}%)")
print(f"Malignant (M): {vc['M']} ({vc['M']/len(df)*100:.1f}%)")
print(f"Ratio B:M = {vc['B']/vc['M']:.2f}:1")

# --- PLOT 5: Class distribution pie + bar ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.pie([vc['B'], vc['M']], labels=[f"Benign\n(n={vc['B']})", f"Malignant\n(n={vc['M']})"],
        colors=[class_colors['B'], class_colors['M']], autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 10})
ax1.set_title('Class Distribution', fontsize=13, fontweight='bold')

ax2.bar(['Benign (B)', 'Malignant (M)'], [vc['B'], vc['M']],
        color=[class_colors['B'], class_colors['M']], edgecolor='white', width=0.5)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title(f'Class Counts (Ratio {vc["B"]/vc["M"]:.2f}:1)', fontsize=13, fontweight='bold')
for i, v in enumerate([vc['B'], vc['M']]):
    ax2.text(i, v + 3, str(v), ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '5_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 5_class_distribution.png")

# =============================================================================
# 6. SUMMARY TABLE (save as CSV for LaTeX import)
# =============================================================================
print("\n" + "=" * 60)
print("6. SAVING SUMMARY TABLE")
print("=" * 60)

summary_rows = []
for cls in classes:
    sub = norm_df[norm_df['class'] == cls]
    n_normal = sub['shapiro_pass'].sum()
    n_features = len(sub)

    corr_matrix = df[df['diagnosis'] == cls][features].dropna().corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    n_high_corr = (upper.stack() > 0.9).sum()
    mean_corr = upper.stack().mean()
    n_skewed = (sub['skewness'].abs() > 1).sum()

    summary_rows.append({
        'Class': f'{cls} ({class_labels[cls]})',
        'N': len(df[df['diagnosis'] == cls]),
        'Normal Features (Shapiro)': f'{n_normal}/{n_features} ({n_normal/n_features*100:.0f}%)',
        'Pairs |r|>0.9': n_high_corr,
        'Mean |r|': f'{mean_corr:.3f}',
        'Highly Skewed (|s|>1)': f'{n_skewed}/{n_features} ({n_skewed/n_features*100:.0f}%)',
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(SAVE_DIR, 'assumption_summary.csv'), index=False)
print(summary_df.to_string(index=False))
print(f"\n  Saved: assumption_summary.csv")

# Also save the full normality results table
norm_export = norm_df[['class', 'feature', 'shapiro_stat', 'shapiro_p',
                        'skewness', 'kurtosis', 'shapiro_pass']].copy()
norm_export.columns = ['Class', 'Feature', 'Shapiro W', 'Shapiro p-value',
                        'Skewness', 'Kurtosis', 'Normal (p>=0.05)']
norm_export.to_csv(os.path.join(SAVE_DIR, 'normality_full_results.csv'), index=False)
print("  Saved: normality_full_results.csv")

print("\n" + "=" * 60)
print("ALL PLOTS AND TABLES SAVED TO:", SAVE_DIR)
print("=" * 60)
