#!/usr/bin/env python3
"""
Plotting script for experimental results.
- clear_only: OA vs sample number with std error bars
- ood: OA vs sample number for different test conditions
- pc_variance: PC1 energy ratio bar chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 150

BASE_DIR = Path(__file__).parent

METHOD_COLORS = {
    'A: Global TruncSVD': '#1f77b4',
    'A*: Global TruncSVD (Centered)': '#ff7f0e',
    'A2: Global TruncSVD (SNV-Pre)': '#2ca02c',
    'A2*: Global TruncSVD (SNV-Pre, Centered)': '#d62728',
    'A3: Global TruncSVD (MSC-Pre)': '#9467bd',
    'A3*: Global TruncSVD (MSC-Pre, Centered)': '#8c564b',
    'F: PLS-DA': '#808080',
}

METHOD_MARKERS = {
    'A: Global TruncSVD': 'o',
    'A*: Global TruncSVD (Centered)': 's',
    'A2: Global TruncSVD (SNV-Pre)': '^',
    'A2*: Global TruncSVD (SNV-Pre, Centered)': 'v',
    'A3: Global TruncSVD (MSC-Pre)': 'D',
    'A3*: Global TruncSVD (MSC-Pre, Centered)': 'p',
    'F: PLS-DA': 'h',
}

EXCLUDE_METHODS = {
    'A_L2: L2 row norm (no SNV)',
    'A_drop1: Drop PC1 (PC2–PC20)',
    'A_drop1_scale: Drop PC1 + Scale (no SNV)',
}

CORE_METHODS = [
    'A: Global TruncSVD',
    'A*: Global TruncSVD (Centered)',
    'A2: Global TruncSVD (SNV-Pre)',
    'A2*: Global TruncSVD (SNV-Pre, Centered)',
]

MSC_VARIANT_METHODS = [
    'A: Global TruncSVD',
    'A3: Global TruncSVD (MSC-Pre)',
    'A3*: Global TruncSVD (MSC-Pre, Centered)',
    'F: PLS-DA',
]

METHOD_STYLE = {
    'A: Global TruncSVD':               {'color': '#1f77b4', 'linestyle': '-',  'linewidth': 3.0, 'marker': 'o', 'markersize': 9, 'label': 'A (Proposed)'},
    'A*: Global TruncSVD (Centered)':   {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 2.2, 'marker': 's', 'markersize': 8, 'label': 'A* (Centered)'},
    'A2: Global TruncSVD (SNV-Pre)':    {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': 2.2, 'marker': '^', 'markersize': 8, 'label': 'A2 (SNV-Pre)'},
    'A2*: Global TruncSVD (SNV-Pre, Centered)': {'color': '#17becf', 'linestyle': ':', 'linewidth': 2.2, 'marker': 'v', 'markersize': 8, 'label': 'A2* (SNV-Pre, Cent.)'},
    'F: PLS-DA':                        {'color': '#808080', 'linestyle': ':',  'linewidth': 2.2, 'marker': 'h', 'markersize': 8, 'label': 'F (PLS-DA)'},
    'A3: Global TruncSVD (MSC-Pre)':    {'color': '#e41a1c', 'linestyle': '--', 'linewidth': 2.2, 'marker': 'D', 'markersize': 8, 'label': 'A3 (MSC-Pre)'},
    'A3*: Global TruncSVD (MSC-Pre, Centered)': {'color': '#984ea3', 'linestyle': '-.', 'linewidth': 2.2, 'marker': 'p', 'markersize': 8, 'label': 'A3* (MSC-Pre, Cent.)'},
}

# Actual training sample counts per ratio (clear-only, 5-fold CV on 193 SWIR samples)
# n_train = max(int(ratio * 154), 6) where 154 ≈ 80% pool, n_classes = 6
RATIO_TO_N_TRAIN = {
    0.01: 6,
    0.10: 15,
    0.15: 23,
    0.20: 30,
    0.30: 46,
    0.50: 77,
    1.00: 154,
}


def plot_clear_only_oa(csv_path, output_path=None, n_repeats=25):
    """
    Two-panel figure for clear-water classification.
    (a) Core comparison: A, A*, A2, F — "ordering matters"
    (b) MSC variants: A, A3, A3* — robustness to scatter correction
    Error bars: 95% CI (1.96 * std / sqrt(n_repeats)).
    """
    df = pd.read_csv(csv_path)
    ci_factor = 1.96 / np.sqrt(n_repeats)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)
    ax_left, ax_right = axes

    panel_configs = [
        (ax_left,  CORE_METHODS,      '(a) Core Comparison'),
        (ax_right, MSC_VARIANT_METHODS, '(b) MSC Variants'),
    ]

    for ax, methods, title in panel_configs:
        for method in methods:
            style = METHOD_STYLE.get(method, {})
            method_data = df[df['method'] == method].sort_values('ratio')
            if method_data.empty:
                continue

            sample_numbers = method_data['ratio'].map(RATIO_TO_N_TRAIN)
            oa_mean = method_data['OA_mean'].values * 100
            oa_ci95 = method_data['OA_std'].values * 100 * ci_factor

            ax.errorbar(
                sample_numbers, oa_mean, yerr=oa_ci95,
                label=style.get('label', method),
                color=style.get('color', '#333'),
                linestyle=style.get('linestyle', '-'),
                linewidth=style.get('linewidth', 1.5),
                marker=style.get('marker', 'o'),
                markersize=style.get('markersize', 6),
                capsize=3, capthick=1.5,
                elinewidth=1.5
            )

        ax.set_xlabel('Number of Training Samples', fontsize=15)
        ax.set_title(title, fontsize=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
        ax.legend(loc='lower right', fontsize=21, framealpha=0.9)

    ax_left.set_ylabel('Overall Accuracy (%)', fontsize=15)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def plot_ood_oa(csv_path, output_path=None,
                conditions=None, n_repeats=25):
    """
    3×2 figure for OOD experiment.
    Rows: ID / OOD-Turbid / OOD-Foamy
    Cols: Core Comparison / MSC Variants
    Error bars: 95% CI when std columns are available.
    """
    df = pd.read_csv(csv_path)

    if conditions is None:
        conditions = ['ID', 'OOD-Turbid', 'OOD-Foamy']

    has_std = all(f'{c}_std' in df.columns for c in conditions)
    ci_factor = 1.96 / np.sqrt(n_repeats) if has_std else None

    n_rows = len(conditions)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows),
                             sharex=True, sharey='row')

    col_methods = [
        (CORE_METHODS,      'Core Comparison'),
        (MSC_VARIANT_METHODS, 'MSC Variants'),
    ]

    col_labels = ['(a)', '(b)']

    for row_idx, cond in enumerate(conditions):
        cond_label = cond.replace('OOD-', '')
        for col_idx, (methods, col_title) in enumerate(col_methods):
            ax = axes[row_idx, col_idx]
            for method in methods:
                style = METHOD_STYLE.get(method, {})
                method_data = df[df['method'] == method].sort_values('ratio')
                if method_data.empty:
                    continue

                sample_numbers = method_data['ratio'].map(RATIO_TO_N_TRAIN)
                col_mean = f'{cond}_mean' if has_std else cond
                oa_mean = method_data[col_mean].values * 100

                if has_std:
                    oa_ci95 = method_data[f'{cond}_std'].values * 100 * ci_factor
                    ax.errorbar(
                        sample_numbers, oa_mean, yerr=oa_ci95,
                        label=style.get('label', method),
                        color=style.get('color', '#333'),
                        linestyle=style.get('linestyle', '-'),
                        linewidth=style.get('linewidth', 1.5),
                        marker=style.get('marker', 'o'),
                        markersize=style.get('markersize', 5),
                        capsize=2.5, capthick=1.2,
                        elinewidth=1.2
                    )
                else:
                    ax.plot(
                        sample_numbers, oa_mean,
                        label=style.get('label', method),
                        color=style.get('color', '#333'),
                        linestyle=style.get('linestyle', '-'),
                        linewidth=style.get('linewidth', 1.5),
                        marker=style.get('marker', 'o'),
                        markersize=style.get('markersize', 5)
                    )

            ax.set_title(f'{col_labels[col_idx]} {cond_label} — {col_title}',
                        fontsize=13)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([0, 105])
            if row_idx == n_rows - 1:
                ax.set_xlabel('Number of Training Samples', fontsize=14)

    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel('Overall Accuracy (%)', fontsize=14)
        axes[row_idx, 0].legend(loc='lower right', fontsize=15, framealpha=0.9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes


def plot_ood_combined(csv_path, output_path=None, n_repeats=25):
    """
    Plot OOD results: all methods on one figure with ID and OOD conditions.
    Error bars represent 95% CI when std columns are available.
    """
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [m for m in df['method'].unique() if m not in EXCLUDE_METHODS]
    conditions = ['ID', 'OOD-Turbid', 'OOD-Foamy']
    linestyles = ['-', '--', ':']

    has_std = all(f'{c}_std' in df.columns for c in conditions)
    ci_factor = 1.96 / np.sqrt(n_repeats) if has_std else None

    for method in methods:
        style = METHOD_STYLE.get(method, {})
        method_data = df[df['method'] == method].sort_values('ratio')
        sample_numbers = method_data['ratio'].map(RATIO_TO_N_TRAIN)
        color = style.get('color', '#333')

        for cond, ls in zip(conditions, linestyles):
            col_mean = f'{cond}_mean' if has_std else cond
            oa_mean = method_data[col_mean].values * 100
            label = f"{style.get('label', method)} ({cond})" if cond != 'ID' else style.get('label', method)
            if has_std:
                oa_ci95 = method_data[f'{cond}_std'].values * 100 * ci_factor
                ax.errorbar(sample_numbers, oa_mean, yerr=oa_ci95, color=color,
                           linestyle=ls, linewidth=1.5, marker='o', markersize=4,
                           label=label, capsize=3, capthick=1.5, elinewidth=1.5)
            else:
                ax.plot(sample_numbers, oa_mean, color=color, linestyle=ls,
                       linewidth=1.5, marker='o', markersize=4, label=label)
    
    ax.set_xlabel('Number of Training Samples', fontsize=16)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=16)
    ax.set_title('OOD Generalization Performance', fontsize=16)
    ax.legend(loc='lower right', fontsize=21, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_pc1_variance(csv_path, output_path=None):
    """
    Plot PC1 energy ratio bar chart for each baseline method.
    """
    df = pd.read_csv(csv_path)
    
    ratios = sorted(df['ratio'].unique())
    methods = [m for m in df['method'].unique() if m not in EXCLUDE_METHODS]
    
    n_methods = len(methods)
    n_ratios = len(ratios)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8 / n_ratios
    x = np.arange(n_methods)
    
    for i, ratio in enumerate(ratios):
        ratio_data = df[df['ratio'] == ratio]
        pc1_values = [ratio_data[ratio_data['method'] == m]['PC1'].values[0] 
                      for m in methods]
        
        offset = (i - n_ratios/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, pc1_values, bar_width, 
                     label=f'ratio={ratio:.0%}', alpha=0.85)
    
    ax.set_xlabel('Method', fontsize=15)
    ax.set_ylabel('PC1 Energy Ratio (%)', fontsize=15)
    ax.set_title('First Principal Component Energy Distribution', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('Global TruncSVD', 'TruncSVD').replace('(', '\n(') 
                        for m in methods], fontsize=12, rotation=0, ha='center')
    ax.legend(loc='upper right', fontsize=21, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_pc1_variance_single_ratio(csv_path, output_path=None, target_ratio=1.0):
    """
    Plot PC1 energy ratio bar chart for a single ratio.
    """
    df = pd.read_csv(csv_path)
    df = df[df['ratio'] == target_ratio]
    
    if df.empty:
        print(f"No data for ratio={target_ratio}")
        return None, None
    
    df = df[~df['method'].isin(EXCLUDE_METHODS)]
    
    methods = df['method'].values
    pc1_values = df['PC1'].values
    pc2_values = df['PC2'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pc1_values, width, label='PC1', color='#1f77b4', alpha=0.85,
                   edgecolor='black', linewidth=1.2, linestyle='-')
    bars2 = ax.bar(x + width/2, pc2_values, width, label='PC2', color='#ff7f0e', alpha=0.85,
                   edgecolor='black', linewidth=1.2, linestyle='--', hatch='//')
    
    ax.set_xlabel('Method', fontsize=15)
    ax.set_ylabel('Energy Ratio (%)', fontsize=15)
    ax.set_title(f'PC1/PC2 Energy Ratio (ratio={target_ratio:.0%})', fontsize=15)
    ax.set_xticks(x)
    
    def method_to_label(m):
        if m == 'A: Global TruncSVD':
            return 'A (proposed)'
        elif m == 'A*: Global TruncSVD (Centered)':
            return 'A*'
        elif m == 'A2: Global TruncSVD (SNV-Pre)':
            return 'A2'
        elif m == 'A2*: Global TruncSVD (SNV-Pre, Centered)':
            return 'A2*'
        elif m == 'A3: Global TruncSVD (MSC-Pre)':
            return 'A3'
        elif m == 'A3*: Global TruncSVD (MSC-Pre, Centered)':
            return 'A3*'
        elif m == 'F: PLS-DA':
            return 'F'
        else:
            return m
    
    ax.set_xticklabels([method_to_label(m) for m in methods], fontsize=12, rotation=0, ha='center')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 105])
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_loading_vectors(csv_path, output_path=None, n_components=20):
    """
    Plot PC1 & PC2 loading vectors from uncentered TruncatedSVD
    on full clear-water samples (ratio=1.00).
    """
    from sklearn.decomposition import TruncatedSVD

    df = pd.read_csv(csv_path, header=None)
    df.columns = ["code", "fraction"] + list(range(350, 2501))
    df["material"] = (df["code"] // 10).astype(int)
    df["water"] = (df["code"] % 10).astype(int)
    df = df[df["material"] != 6]

    band_range = (940, 1680)
    all_wvl = np.arange(350, 2501, dtype=float)
    band_mask = (all_wvl >= band_range[0]) & (all_wvl <= band_range[1])
    wavelengths = all_wvl[band_mask]
    band_cols = list(all_wvl[band_mask].astype(int))

    X = df[band_cols].values.astype(np.float32)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)
    V = svd.components_

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(wavelengths, V[0, :], label='PC1 Loading', color='black',
            linewidth=1.8, linestyle='-')
    ax.plot(wavelengths, V[1, :], label='PC2 Loading', color='#d62728',
            linewidth=1.4, linestyle='--')

    ax.axvspan(1340, 1460, alpha=0.10, color='gray', label='Water absorption gap')

    ax.set_xlabel('Wavelength (nm)', fontsize=15)
    ax.set_ylabel('Loading Value (Arbitrary Units)', fontsize=15)
    ax.set_title('Loading Vectors of the First Two Uncentered PCs', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def plot_textile_pc1_verification(csv_path, output_path=None):
    """
    PC1 energy ratio verification on textile SWIR dataset.
    Computes uncentered SVD on full samples and reports r1.
    Also produces a comparison bar chart with the plastic dataset result.
    """
    from sklearn.decomposition import TruncatedSVD

    df = pd.read_csv(csv_path, sep=';')

    wvl_cols = [c for c in df.columns
                if c.replace('.', '').isdigit() and 1000 <= float(c) <= 1680]
    wavelengths = np.array([float(c) for c in wvl_cols])
    X = df[wvl_cols].values.astype(np.float32)

    print(f"Textile dataset: {X.shape[0]} samples x {X.shape[1]} bands "
          f"({wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm)")

    svd = TruncatedSVD(n_components=20, random_state=42)
    svd.fit(X)
    sv_sq = svd.singular_values_ ** 2
    total_energy = np.sum(X ** 2)
    r1_textile = sv_sq[0] / total_energy * 100
    r2_textile = sv_sq[1] / total_energy * 100

    print(f"  PC1 energy ratio (r1): {r1_textile:.2f}%")
    print(f"  PC2 energy ratio (r2): {r2_textile:.2f}%")
    print(f"  PC1+PC2: {r1_textile + r2_textile:.2f}%")

    V = svd.components_

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.plot(wavelengths, V[0, :], label='PC1 Loading', color='black',
            linewidth=1.8, linestyle='-')
    ax.plot(wavelengths, V[1, :], label='PC2 Loading', color='#d62728',
            linewidth=1.4, linestyle='--')
    ax.set_xlabel('Wavelength (nm)', fontsize=15)
    ax.set_ylabel('Loading Value (Arbitrary Units)', fontsize=15)
    ax.set_title('Loading Vectors (Textile SWIR)', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax = axes[1]
    plastic_csv = BASE_DIR / 'pc_variance_summary.csv'
    r1_plastic = None
    if plastic_csv.exists():
        pc_df = pd.read_csv(plastic_csv)
        pc_df = pc_df[(pc_df['ratio'] == 1.0) &
                       (pc_df['method'] == 'A: Global TruncSVD')]
        if not pc_df.empty:
            r1_plastic = pc_df['PC1'].values[0]

    datasets = ['Textile\n(SWIR 1000–1680nm)']
    r1_values = [r1_textile]
    colors = ['#2ca02c']
    if r1_plastic is not None:
        datasets.append('Plastic\n(SWIR 940–1680nm)')
        r1_values.append(r1_plastic)
        colors.append('#1f77b4')

    bars = ax.bar(datasets, r1_values, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, r1_values):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC1 Energy Ratio (%)', fontsize=15)
    ax.set_title('Cross-Domain r\u2081 Comparison', fontsize=15)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, axes, r1_textile


if __name__ == '__main__':
    
    clear_csv = BASE_DIR / 'csv_results_summary.csv'
    ood_csv = BASE_DIR / 'ood_water_results_summary.csv'
    pc_csv = BASE_DIR / 'pc_variance_summary.csv'
    
    print("=" * 60)
    print("Plotting Clear-Only Results")
    print("=" * 60)
    if clear_csv.exists():
        plot_clear_only_oa(clear_csv, BASE_DIR / 'fig_clear_only_oa.png')
    else:
        print(f"File not found: {clear_csv}")
    
    print("\n" + "=" * 60)
    print("Plotting OOD Results")
    print("=" * 60)
    if ood_csv.exists():
        plot_ood_oa(ood_csv, BASE_DIR / 'fig_ood_oa_separate.png',
                   conditions=['ID', 'OOD-Turbid', 'OOD-Foamy'])
    else:
        print(f"File not found: {ood_csv}")
    
    print("\n" + "=" * 60)
    print("Plotting PC Variance Results")
    print("=" * 60)
    if pc_csv.exists():
        plot_pc1_variance(pc_csv, BASE_DIR / 'fig_pc1_variance_all.png')
        plot_pc1_variance_single_ratio(pc_csv, BASE_DIR / 'fig_pc1_variance_ratio100.png', 
                                       target_ratio=1.0)
    else:
        print(f"File not found: {pc_csv}")
    
    print("\n" + "=" * 60)
    print("Plotting Loading Vectors")
    print("=" * 60)
    data_csv = BASE_DIR / '1' / 'spectradictionary.csv'
    if data_csv.exists():
        plot_loading_vectors(data_csv, BASE_DIR / 'fig_loading_vectors.png')
    else:
        print(f"File not found: {data_csv}")
    
    print("\n" + "=" * 60)
    print("Textile PC1 Energy Ratio Verification")
    print("=" * 60)
    textile_csv = BASE_DIR / '1' / 'swir_mean_spectra.csv'
    if textile_csv.exists():
        plot_textile_pc1_verification(textile_csv,
                                      BASE_DIR / 'fig_textile_pc1_verification.png')
    else:
        print(f"File not found: {textile_csv}")
    
    print("\nDone!")
