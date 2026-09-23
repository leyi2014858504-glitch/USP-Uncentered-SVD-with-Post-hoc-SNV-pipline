#!/usr/bin/env python3
"""
run_spectra_figures.py
======================
Three reviewer-requested figures for the main plastic dataset
(Clear water, 193 samples, 940-1680 nm, 6 classes):

  Fig 1 — fig_spectra_overlay.png
      Raw reflectance spectra overlay, colored by class.
      All samples (thin lines) + class mean (bold line).
      (Reviewer Fig.2a/8a style — the basic spectral display.)

  Fig 2 — fig_spectra_preprocessing.png
      3-panel before/after preprocessing comparison: Raw | SNV | USP.
      Mean ± std shaded band per class. Shows USP's scatter removal.

  Fig 3 — fig_confusion_matrix.png
      USP confusion matrix at ratio=1.0, 6-class, pooled over
      5-fold CV × 5 seeds (25 evaluations), row-normalized (recall).
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"
N_DIM = 20
SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 5

MATERIAL_NAMES = {1: "PET", 2: "HDPE", 3: "LDPE",
                  4: "PP", 5: "EPSF", 7: "Weathered"}
CLASS_ORDER = [1, 2, 3, 4, 5, 7]   # numeric code order
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:6]
COLOR_MAP = {cls: COLORS[i] for i, cls in enumerate(CLASS_ORDER)}


# ───────────────────────── Data ─────────────────────────

def load_clear_plastic(csv_path, band_range=(940, 1680)):
    """Load Clear-water river plastics (same as usp_pipeline.load_river_csv)."""
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["code", "fraction"] + list(range(350, 2501))
    df["material"] = (df["code"] // 10).astype(int)
    df["water"] = (df["code"] % 10).astype(int)
    df = df[df["material"] != 6]            # exclude Mix
    df = df[df["water"] == 0]               # Clear only
    band_cols = list(range(band_range[0], band_range[1] + 1))
    wvl = np.array(band_cols, dtype=float)
    X = df[band_cols].values.astype(np.float32)
    y = df["material"].values.astype(np.int32)
    return X, y, wvl


# ───────────────────────── Preprocessing ─────────────────────────

def snv(X, eps=1e-8):
    """Row-wise (per-sample) standard normal variate."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def usp_reconstruct(X, n_dim=N_DIM):
    """USP-corrected spectra reconstructed to wavelength domain.

    Raw → uncentered TruncatedSVD(k) → post-SNV in PC space → reconstruct.
    """
    n_comp = max(1, min(n_dim, X.shape[0] - 1, X.shape[1]))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z = svd.fit_transform(X)          # (n, k) uncentered scores
    Z_snv = snv(Z)                    # post-SNV in PC space
    X_usp = Z_snv @ svd.components_   # (n, p) reconstruct to wavelength
    return X_usp.astype(np.float32)


def usp_predict(Xtr, Xte, ytr):
    """Full USP pipeline → predictions (matches run_baselines_kfold)."""
    n_comp = max(1, min(N_DIM, Xtr.shape[0] - 1, Xtr.shape[1]))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Ztr = svd.fit_transform(Xtr)
    Zte = svd.transform(Xte)
    Ztr = snv(Ztr)
    Zte = snv(Zte)
    sc = StandardScaler().fit(Ztr)
    clf = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
    clf.fit(sc.transform(Ztr), ytr)
    return clf.predict(sc.transform(Zte))


# ───────────────────────── Figure 1: Raw overlay ─────────────────────────

def fig_raw_overlay(X, y, wvl):
    """All samples (thin) + class mean (bold), colored by class."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cls in CLASS_ORDER:
        mask = y == cls
        if mask.sum() == 0:
            continue
        Xc = X[mask]
        c = COLOR_MAP[cls]
        name = MATERIAL_NAMES[cls]
        # thin lines: all samples
        for i in range(Xc.shape[0]):
            ax.plot(wvl, Xc[i], color=c, alpha=0.25, lw=0.6)
        # bold mean line
        ax.plot(wvl, Xc.mean(axis=0), color=c, lw=2.4,
                label=f"{name} (n={mask.sum()})")
    ax.set_xlabel("Wavelength (nm)", fontsize=13)
    ax.set_ylabel("Reflectance", fontsize=13)
    ax.set_title("Raw Reflectance Spectra by Polymer Class", fontsize=14,
                 fontweight="bold")
    ax.set_xlim(wvl[0], wvl[-1])
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, ncol=2, loc="best", framealpha=0.9)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = BASE_DIR / "fig_spectra_overlay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ───────────────────────── Figure 2: Preprocessing 3-panel ─────────────────────────

def _plot_mean_std(ax, X, y, wvl, title, ylabel):
    """Mean ± std shaded band per class."""
    for cls in CLASS_ORDER:
        mask = y == cls
        if mask.sum() == 0:
            continue
        Xc = X[mask]
        m = Xc.mean(axis=0)
        s = Xc.std(axis=0)
        c = COLOR_MAP[cls]
        name = MATERIAL_NAMES[cls]
        ax.plot(wvl, m, color=c, lw=1.8, label=name)
        ax.fill_between(wvl, m - s, m + s, color=c, alpha=0.18, lw=0)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(wvl[0], wvl[-1])
    ax.grid(alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=10)


def fig_preprocessing(X, y, wvl):
    """3-panel: Raw | SNV | USP, mean ± std per class."""
    X_snv = snv(X)
    X_usp = usp_reconstruct(X)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    _plot_mean_std(axes[0], X, y, wvl, "(a) Raw", "Reflectance")
    _plot_mean_std(axes[1], X_snv, y, wvl, "(b) SNV", "Reflectance (a.u.)")
    _plot_mean_std(axes[2], X_usp, y, wvl, "(c) USP", "Reflectance (a.u.)")
    # single shared legend on last panel
    axes[2].legend(fontsize=9, ncol=2, loc="best", framealpha=0.9)
    fig.tight_layout()
    out = BASE_DIR / "fig_spectra_preprocessing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # quantitative tightening
    raw_std = np.mean([X[y == c].std(axis=0).mean()
                       for c in CLASS_ORDER if (y == c).sum() > 0])
    snv_std = np.mean([X_snv[y == c].std(axis=0).mean()
                       for c in CLASS_ORDER if (y == c).sum() > 0])
    usp_std = np.mean([X_usp[y == c].std(axis=0).mean()
                       for c in CLASS_ORDER if (y == c).sum() > 0])
    print(f"  Intra-class std: Raw={raw_std:.4f}  SNV={snv_std:.4f}  "
          f"USP={usp_std:.4f}  (USP reduction: {(1-usp_std/raw_std)*100:.1f}%)")


# ───────────────────────── Figure 3: Confusion matrix ─────────────────────────

def fig_confusion_matrix(X, y):
    """USP @ ratio=1.0, pooled over 5-fold CV × 5 seeds, row-normalized."""
    y_true_all, y_pred_all = [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            y_pred = usp_predict(Xtr, Xte, ytr)
            y_true_all.extend(yte.tolist())
            y_pred_all.extend(y_pred.tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    oa = accuracy_score(y_true_all, y_pred_all)

    # row-normalized (recall) confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=CLASS_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = [MATERIAL_NAMES[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            v = cm_norm[i, j]
            txt = f"{v*100:.1f}%"
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(labels, fontsize=11, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted class", fontsize=13)
    ax.set_ylabel("True class", fontsize=13)
    ax.set_title(f"USP Confusion Matrix (ratio=1.0, pooled OA={oa*100:.2f}%)",
                 fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalized)", fontsize=11)
    fig.tight_layout()
    out = BASE_DIR / "fig_confusion_matrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  Pooled OA (25 evals): {oa*100:.2f}%")
    print("  Per-class recall:")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {MATERIAL_NAMES[cls]:<10}: {cm_norm[i, i]*100:.2f}%")


# ───────────────────────── Main ─────────────────────────

def run():
    csv_path = DATA_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        csv_path = BASE_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        print("ERROR: spectradictionary.csv not found")
        return

    X, y, wvl = load_clear_plastic(csv_path)
    print(f"[Data] {len(y)} samples | {X.shape[1]} bands "
          f"({wvl[0]:.0f}-{wvl[-1]:.0f} nm) | {len(np.unique(y))} classes\n")

    print("=" * 60)
    print("Fig 1: Raw reflectance overlay")
    print("=" * 60)
    fig_raw_overlay(X, y, wvl)

    print("\n" + "=" * 60)
    print("Fig 2: Preprocessing comparison (Raw | SNV | USP)")
    print("=" * 60)
    fig_preprocessing(X, y, wvl)

    print("\n" + "=" * 60)
    print("Fig 3: USP confusion matrix (ratio=1.0)")
    print("=" * 60)
    fig_confusion_matrix(X, y)


if __name__ == "__main__":
    run()
