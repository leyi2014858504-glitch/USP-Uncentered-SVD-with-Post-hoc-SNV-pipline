#!/usr/bin/env python3
"""
run_hyperparam_sweep.py
=======================
Reviewer-requested hyperparameter sensitivity sweep on the main plastic
dataset (Clear water, 193 samples, 940-1680 nm, 6 classes).

Goal: show the USP-vs-SNV comparison is robust across a reasonable SVM
hyperparameter grid (not an artifact of the default C=1).

Design (user-confirmed):
  - Approach : sensitivity landscape (report full grid, no selection bias)
  - Methods  : USP + SNV
  - Eval     : Normal OA only (BandShift covered elsewhere)
  - Classifiers: LinearSVM + RBF-SVM (no RF)
  - Protocol : 5-fold CV x 5 seeds = 25 evals per grid point, ratio=1.0

Grid:
  - LinearSVM : C in [0.01, 0.1, 1, 10, 100]                  (1-D)
  - RBF-SVM   : C in [0.1, 1, 10, 100] x gamma in [1e-3,1e-2,1e-1,1,10]  (2-D)

Reuses pipeline functions from run_baselines_kfold.py (imported, not
duplicated). That module's make_clf/pipeline_* now accept C/gamma.

Outputs:
  results_hyperparam_raw.csv        - per seed/fold
  results_hyperparam_summary.csv    - mean/std per (method,clf,C,gamma)
  fig_hyperparam_linear.png         - OA & macro-F1 vs C (LinearSVM)
  fig_hyperparam_rbf.png            - C-gamma heatmaps (RBF-SVM, USP | SNV)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from run_baselines_kfold import (
    load_clear_plastic, snv, pipeline_usp, pipeline_traditional, evaluate,
    SEEDS, N_SPLITS, DATA_DIR, CLF_LABELS,
)

BASE_DIR = Path(__file__).parent

# Methods: (name, pipeline_fn, uses_raw) — SNV feeds preprocessed input,
# USP feeds raw input (pipeline does its own SVD first).
METHODS = [
    ("USP", pipeline_usp, True),
    ("SNV", pipeline_traditional, False),
]
CLASSIFIERS = ["svm", "rbf_svm"]

# Grids
C_LINEAR = [0.01, 0.1, 1, 10, 100]
C_RBF = [0.1, 1, 10, 100]
GAMMA_RBF = [0.001, 0.01, 0.1, 1, 10]
DEFAULT_C = 1.0


# ───────────────────────── Sweep ─────────────────────────

def run():
    csv_path = DATA_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        csv_path = BASE_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        print("ERROR: spectradictionary.csv not found")
        return

    X, y = load_clear_plastic(csv_path)
    n_cls = len(np.unique(y))
    print(f"[Data] {len(y)} samples | {n_cls} classes | {X.shape[1]} bands")
    print(f"[Protocol] 5-fold CV x {len(SEEDS)} seeds = "
          f"{N_SPLITS * len(SEEDS)} evals/grid point | ratio=1.0 | Normal only")
    print(f"[Grid] LinearSVM C={C_LINEAR}")
    print(f"[Grid] RBF-SVM C={C_RBF} x gamma={GAMMA_RBF}")

    records = []
    eval_count = 0
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=seed)
        for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr].astype(np.float32), X[te].astype(np.float32)
            ytr, yte = y[tr], y[te]
            Xtr_snv, Xte_snv = snv(Xtr), snv(Xte)

            for method_name, pipe_fn, uses_raw in METHODS:
                Xtr_in = Xtr if uses_raw else Xtr_snv
                Xte_in = Xte if uses_raw else Xte_snv

                for clf_type in CLASSIFIERS:
                    if clf_type == "svm":
                        grid = [(c, "scale") for c in C_LINEAR]
                    else:
                        grid = [(c, g) for c in C_RBF for g in GAMMA_RBF]

                    for C, gamma in grid:
                        y_pred, *_ = pipe_fn(Xtr_in, Xte_in, ytr,
                                             clf_type, C=C, gamma=gamma)
                        m = evaluate(yte, y_pred)
                        records.append({
                            "method": method_name,
                            "classifier": clf_type,
                            "C": C, "gamma": gamma,
                            "seed": seed, "fold": fold_idx,
                            **m,
                        })
                        eval_count += 1
        print(f"  seed {seed} done ({eval_count} evals)")

    df = pd.DataFrame(records)
    out_raw = BASE_DIR / "results_hyperparam_raw.csv"
    df.to_csv(out_raw, index=False, float_format="%.6f")
    print(f"\nSaved raw: {out_raw} ({len(df)} rows)")

    # ─── Summary ───
    summary = df.groupby(["method", "classifier", "C", "gamma"]).agg(
        OA_mean=("OA", "mean"), OA_std=("OA", "std"),
        macro_F1_mean=("macro_F1", "mean"), macro_F1_std=("macro_F1", "std"),
        kappa_mean=("kappa", "mean"),
    ).reset_index()
    out_sum = BASE_DIR / "results_hyperparam_summary.csv"
    summary.to_csv(out_sum, index=False, float_format="%.6f")
    print(f"Saved summary: {out_sum} ({len(summary)} rows)")

    # ─── Robustness check ───
    print("\n" + "=" * 80)
    print("Robustness: USP vs SNV across grid")
    print("=" * 80)
    for clf_type in CLASSIFIERS:
        sub = summary[summary["classifier"] == clf_type]
        # pivot: index=grid combo, columns=method, values=OA_mean
        piv = sub.pivot_table(index=["C", "gamma"],
                              columns="method", values="OA_mean")
        if "USP" in piv.columns and "SNV" in piv.columns:
            win = int((piv["USP"] > piv["SNV"]).sum())
            tot = len(piv)
            print(f"  {CLF_LABELS[clf_type]:<10}: USP > SNV on "
                  f"{win}/{tot} grid points")

        # default C=1 vs best, per method
        for method in ["USP", "SNV"]:
            msub = sub[sub["method"] == method]
            default_rows = msub[msub["C"] == DEFAULT_C]
            if clf_type == "svm":
                d_oa = default_rows["OA_mean"].values[0] * 100
                d_txt = f"C=1 (default) OA={d_oa:.2f}%"
            else:
                lo = default_rows["OA_mean"].min() * 100
                hi = default_rows["OA_mean"].max() * 100
                d_txt = f"C=1 across gamma OA=[{lo:.2f}, {hi:.2f}]%"
            best_row = msub.loc[msub["OA_mean"].idxmax()]
            best_oa = best_row["OA_mean"] * 100
            gtxt = f", gamma={best_row['gamma']:g}" if clf_type == "rbf_svm" else ""
            print(f"    {method:<4} @ {CLF_LABELS[clf_type]:<10} "
                  f"{d_txt} | best OA={best_oa:.2f}% "
                  f"(C={best_row['C']:g}{gtxt})")

    # ─── Figures ───
    plot_linear(summary)
    plot_rbf(summary)
    print("\nDone.")


# ───────────────────────── Figures ─────────────────────────

COLORS = {"USP": "#1f77b4", "SNV": "#ff7f0e"}


def plot_linear(summary):
    """OA & macro-F1 vs C for LinearSVM (USP vs SNV)."""
    sub = summary[summary["classifier"] == "svm"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = [("OA_mean", "OA_std", "Overall Accuracy"),
               ("macro_F1_mean", "macro_F1_std", "Macro-F1")]

    for ax, (mcol, scol, title) in zip(axes, metrics):
        for method in ["USP", "SNV"]:
            d = sub[sub["method"] == method].sort_values("C")
            C = d["C"].values
            mean = d[mcol].values * 100
            std = d[scol].values * 100
            ax.plot(C, mean, "-o", color=COLORS[method], label=method,
                    linewidth=2, markersize=6)
            ax.fill_between(C, mean - std, mean + std, color=COLORS[method],
                            alpha=0.15)
        ax.set_xscale("log")
        ax.axvline(DEFAULT_C, color="gray", linestyle="--", linewidth=1,
                   alpha=0.7, label="default C=1")
        ax.set_xlabel("C (regularization, log scale)", fontsize=12)
        ax.set_ylabel(f"{title} (%)", fontsize=12)
        ax.set_title(f"LinearSVM — {title} vs C", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hyperparameter Sensitivity — LinearSVM (5-fold CV x 5 seeds)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = BASE_DIR / "fig_hyperparam_linear.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


def plot_rbf(summary):
    """C-gamma heatmaps for RBF-SVM (USP | SNV)."""
    sub = summary[summary["classifier"] == "rbf_svm"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, method in zip(axes, ["USP", "SNV"]):
        d = sub[sub["method"] == method]
        piv = d.pivot_table(index="gamma", columns="C", values="OA_mean")
        # order rows: small gamma (top) -> large gamma (bottom) reversed for log feel
        piv = piv.sort_index(ascending=False)
        im = ax.imshow(piv.values * 100, aspect="auto", cmap="viridis",
                       vmin=0, vmax=100)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=10)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{g:g}" for g in piv.index], fontsize=10)
        ax.set_xlabel("C", fontsize=12)
        ax.set_ylabel("gamma", fontsize=12)
        ax.set_title(f"RBF-SVM — {method} (mean OA %)", fontsize=13,
                     fontweight="bold")
        # annotate cells
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j] * 100
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if v < 60 else "black")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Mean OA (%)", fontsize=11)
    fig.suptitle("Hyperparameter Sensitivity — RBF-SVM (5-fold CV x 5 seeds)",
                 fontsize=14, fontweight="bold", y=1.02)
    out = BASE_DIR / "fig_hyperparam_rbf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    run()
