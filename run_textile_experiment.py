#!/usr/bin/env python3
"""
run_textile_experiment.py
========================
Cross-domain validation: run the same classification pipeline on the
OpenTextile SWIR dataset to prove the effectiveness of post-SNV.

Data: swir_mean_spectra.csv + ground_truth_final.csv
  - 71 samples, filter out 3 unknown + 8 outlier → ~57 usable (exact count depends on CSV intersection)
  - Take dominant fiber as label
  - Band range: 1000–1680 nm (overlap with plastic dataset)
  - Only 3 core methods: A, A2 (SNV-Pre), F (PLS-DA)
  - Leave-One-Out CV (sample-level, ~57 folds)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, cohen_kappa_score)
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"

FIBER_NAMES = {
    0: "Cotton", 1: "Polyester", 2: "Wool",
    3: "Polyamide", 4: "Viscose", 5: "Lyocell/Modacrylic"
}

CORE_METHODS = ["A", "A2", "A_noSNV", "F"]


def load_textile_data(spectra_path, gt_path, band_range=(1000, 1680),
                      min_dominant_frac=0.0, exclude_unknown=True,
                      exclude_outlier=True):
    spectra_df = pd.read_csv(spectra_path, sep=';', index_col=0)
    gt_df = pd.read_csv(gt_path, sep=';', index_col=0)

    common = spectra_df.index.intersection(gt_df.index)
    spectra_df = spectra_df.loc[common]
    gt_df = gt_df.loc[common]

    if exclude_unknown:
        mask_unk = gt_df["Unknown or outlier?"].fillna("") != "unknown"
        spectra_df = spectra_df[mask_unk]
        gt_df = gt_df.loc[spectra_df.index]

    if exclude_outlier:
        mask_out = gt_df["Unknown or outlier?"].fillna("") != "outlier"
        spectra_df = spectra_df[mask_out]
        gt_df = gt_df.loc[spectra_df.index]

    wvl_cols = [c for c in spectra_df.columns
                if c.replace('.', '').isdigit()
                and band_range[0] <= float(c) <= band_range[1]]
    wavelengths = np.array([float(c) for c in wvl_cols])
    X = spectra_df[wvl_cols].values.astype(np.float32)

    fiber_cols = ["Polyamide [%]", "Viscose [%]", "Lyocell [%]",
                  "Modacrylic [%]", "Polyacrylic [%]", "Wool [%]",
                  "Carbon fibre [%]", "Polyurethane [%]", "Elastane [%]",
                  "Polyester [%]", "Cotton [%]"]

    labels = []
    for idx in spectra_df.index:
        row = gt_df.loc[idx]
        comps = {}
        for fc in fiber_cols:
            if fc in row.index and pd.notna(row[fc]):
                name = fc.replace(" [%]", "").replace("Carbon fibre", "Wool")
                if name == "Modacrylic" or name == "Lyocell":
                    name = "Lyocell/Modacrylic"
                comps[name] = comps.get(name, 0.0) + float(row[fc])

        if not comps:
            labels.append(-1)
            continue

        dominant = max(comps, key=comps.get)
        if comps[dominant] < min_dominant_frac * 100:
            labels.append(-1)
        else:
            label_map = {"Cotton": 0, "Polyester": 1, "Wool": 2,
                         "Polyamide": 3, "Viscose": 4,
                         "Lyocell/Modacrylic": 5}
            labels.append(label_map.get(dominant, -1))

    y = np.array(labels, dtype=np.int32)
    valid = y >= 0
    X = X[valid]
    y = y[valid]
    sample_ids = spectra_df.index[valid]

    print(f"[Textile] {len(y)} samples | {len(np.unique(y))} fiber types | "
          f"{X.shape[1]} bands ({wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm)")
    for cls in np.unique(y):
        print(f"  {FIBER_NAMES.get(int(cls), int(cls))}: {(y == cls).sum()} samples")

    return X, y, wavelengths, sample_ids


def snv(X, eps=1e-8):
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def method_a(Xtr, Xte, n_dim=20):
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    svd.fit(Xtr)
    return svd.transform(Xtr), svd.transform(Xte), svd


def method_a2(Xtr, Xte, n_dim=20):
    Xtr_s = snv(Xtr)
    Xte_s = snv(Xte)
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    svd.fit(Xtr_s)
    return svd.transform(Xtr_s), svd.transform(Xte_s), svd


def method_a_nosnv(Xtr, Xte, n_dim=20):
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    svd.fit(Xtr)
    return svd.transform(Xtr), svd.transform(Xte), svd


def fit_plsda(X, y, n_comp=20):
    from sklearn.cross_decomposition import PLSRegression
    classes = np.unique(y)
    Y_dummy = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        Y_dummy[y == c, i] = 1
    n_comp = min(n_comp, len(classes) - 1, X.shape[0] - 1, X.shape[1])
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X, Y_dummy)
    return pls, classes


def predict_plsda(pls, classes, X):
    Y_pred = pls.predict(X)
    return classes[np.argmax(Y_pred, axis=1)]


def make_clf():
    return SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42,
               decision_function_shape='ovo')


def compute_metrics(y_true, y_pred):
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "AA": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def run_textile_loo(X, y, n_dim=20):
    loo = LeaveOneOut()
    records = []

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        results = {}

        # A: Global TruncSVD + post-SNV
        Xtr_f, Xte_f, svd_a = method_a(Xtr, Xte, n_dim)
        Xtr_f = snv(Xtr_f)
        Xte_f = snv(Xte_f)
        sc = StandardScaler()
        clf = make_clf()
        clf.fit(sc.fit_transform(Xtr_f), ytr)
        y_pred = clf.predict(sc.transform(Xte_f))
        results["A"] = compute_metrics(yte, y_pred)

        # A2: SNV-Pre + TruncSVD (no post-SNV)
        Xtr_f2, Xte_f2, svd_a2 = method_a2(Xtr, Xte, n_dim)
        sc2 = StandardScaler()
        clf2 = make_clf()
        clf2.fit(sc2.fit_transform(Xtr_f2), ytr)
        y_pred2 = clf2.predict(sc2.transform(Xte_f2))
        results["A2"] = compute_metrics(yte, y_pred2)

        # A_noSNV: Global TruncSVD without any SNV
        Xtr_f3, Xte_f3, svd_a3 = method_a_nosnv(Xtr, Xte, n_dim)
        sc3 = StandardScaler()
        clf3 = make_clf()
        clf3.fit(sc3.fit_transform(Xtr_f3), ytr)
        y_pred3 = clf3.predict(sc3.transform(Xte_f3))
        results["A_noSNV"] = compute_metrics(yte, y_pred3)

        # F: PLS-DA
        try:
            pls, classes = fit_plsda(Xtr, ytr, n_comp=min(n_dim, len(np.unique(ytr)) - 1))
            y_pred_f = predict_plsda(pls, classes, Xte)
            results["F"] = compute_metrics(yte, y_pred_f)
        except Exception:
            results["F"] = {m: np.nan for m in ["OA", "AA", "macro_F1", "kappa"]}

        for method_name in CORE_METHODS:
            records.append({
                "fold": fold_idx,
                "method": method_name,
                **results[method_name]
            })

        if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
            print(f"  Fold {fold_idx+1}/{len(y)}: "
                  f"A={results['A']['OA']:.0f} "
                  f"A2={results['A2']['OA']:.0f} "
                  f"A_noSNV={results['A_noSNV']['OA']:.0f} "
                  f"F={results['F']['OA']:.0f}")

    return pd.DataFrame(records)


def summarize_results(df):
    summary = df.groupby("method").agg(
        OA_mean=("OA", "mean"),
        OA_std=("OA", "std"),
        AA_mean=("AA", "mean"),
        macro_F1_mean=("macro_F1", "mean"),
        macro_F1_std=("macro_F1", "std"),
        kappa_mean=("kappa", "mean"),
    ).reset_index()

    method_order = {m: i for i, m in enumerate(CORE_METHODS)}
    summary["_order"] = summary["method"].map(method_order)
    summary = summary.sort_values("_order").drop(columns=["_order"])

    return summary


METHOD_LABELS = {
    "A":        "A (Proposed: TruncSVD + post-SNV)",
    "A2":       "A2 (SNV-Pre + TruncSVD)",
    "A_noSNV":  "A (TruncSVD, no SNV)",
    "F":        "F (PLS-DA)",
}

METHOD_COLORS = {
    "A":        "#1f77b4",
    "A2":       "#2ca02c",
    "A_noSNV":  "#ff7f0e",
    "F":        "#808080",
}


def plot_textile_results(summary, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = summary["method"].values
    oa_vals = summary["OA_mean"].values * 100
    oa_stds = summary["OA_std"].values * 100
    f1_vals = summary["macro_F1_mean"].values * 100

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, oa_vals, width, label='OA',
                   color=[METHOD_COLORS.get(m, '#333') for m in methods],
                   alpha=0.85, edgecolor='black', linewidth=1.2, linestyle='-')
    bars2 = ax.bar(x + width/2, f1_vals, width, label='macro-F1',
                   color=[METHOD_COLORS.get(m, '#333') for m in methods],
                   alpha=0.55, edgecolor='black', linewidth=1.2, linestyle='--',
                   hatch='//')

    for bar, val in zip(bars1, oa_vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, f1_vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Performance (%)', fontsize=15)
    ax.set_title('Textile Fiber Classification (LOO-CV)', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 105])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


if __name__ == "__main__":
    spectra_csv = DATA_DIR / "swir_mean_spectra.csv"
    gt_csv = DATA_DIR / "ground_truth_final.csv"

    X, y, wvl, sample_ids = load_textile_data(
        spectra_csv, gt_csv,
        band_range=(1000, 1680),
        exclude_unknown=True,
        exclude_outlier=True,
    )

    print(f"\nRunning Leave-One-Out CV on {len(y)} samples ...")
    df_results = run_textile_loo(X, y, n_dim=20)

    summary = summarize_results(df_results)
    print("\n" + "=" * 60)
    print("Textile Classification Results (LOO-CV)")
    print("=" * 60)
    print(summary.to_string(index=False, float_format="%.3f"))

    csv_out = BASE_DIR / "textile_results_summary.csv"
    summary.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_out}")

    fig_out = BASE_DIR / "fig_textile_classification.png"
    plot_textile_results(summary, fig_out)

    print("\nDone!")
