#!/usr/bin/env python3
"""
run_pc2_extrema.py
==================
Quantify PC2 loading-vector extrema (peaks & troughs) for the reviewer's
band-assignment request: instead of "PC2 looks like it has peaks in the
C-H region", report the exact wavelengths and loading values.

Datasets (uncentered TruncatedSVD, k=20, same as USP pipeline):
  1. Original plastic dataset (Clear water, 193 samples, 940-1680 nm)
  2. Generalization dataset (pooled, 6 polymers, 940-1680 nm)

Method: scipy.signal.find_peaks on PC2 (and -PC2 for troughs), with a
prominence threshold relative to the loading range; extrema are merged,
sorted by prominence, and exported.

Outputs:
  results_pc2_extrema.csv
  fig_pc2_annotated.png  (PC2 curve with labeled extrema)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import TruncatedSVD

from run_log_preprocessing import load_data
from run_generalization_experiment import (
    load_generalization_data, DATA_DIR as GEN_DATA_DIR)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"
N_DIM = 20
BAND = (940, 1680)


def pc2_loadings(X):
    svd = TruncatedSVD(n_components=N_DIM, random_state=42)
    svd.fit(X)
    return svd.components_[1], svd.explained_variance_ratio_


def extract_extrema(wvl, v2, min_prom_frac=0.08):
    """Find peaks/troughs of PC2 loading; return DataFrame sorted by prominence."""
    rng = v2.max() - v2.min()
    prom_min = min_prom_frac * rng
    pk, pk_prop = find_peaks(v2, prominence=prom_min)
    tr, tr_prop = find_peaks(-v2, prominence=prom_min)
    rows = []
    for i, p in zip(pk, pk_prop["prominences"]):
        rows.append({"wavelength_nm": float(wvl[i]),
                     "loading": float(v2[i]), "type": "peak",
                     "prominence": float(p)})
    for i, p in zip(tr, tr_prop["prominences"]):
        rows.append({"wavelength_nm": float(wvl[i]),
                     "loading": float(v2[i]), "type": "trough",
                     "prominence": float(p)})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("prominence", ascending=False).reset_index(drop=True)
    return df


def orig_csv():
    p = DATA_DIR / "spectradictionary.csv"
    return p if p.exists() else BASE_DIR / "spectradictionary.csv"


def gen_csv():
    p = GEN_DATA_DIR / "__Mean_Spectra_Reflectance.csv"
    return p if p.exists() else BASE_DIR / "__Mean_Spectra_Reflectance.csv"


def main():
    all_rows = []

    # ── Dataset 1: original Clear ──
    X, y = load_data(str(orig_csv()), water_filter=0)
    wvl = np.arange(BAND[0], BAND[1] + 1, dtype=float)[: X.shape[1]]
    v2, evr = pc2_loadings(X)
    df1 = extract_extrema(wvl, v2)
    df1.insert(0, "dataset", "Original_Clear")
    print(f"\n[Original Clear] PC1={evr[0]*100:.2f}%  PC2={evr[1]*100:.2f}%")
    print(df1.to_string(index=False))

    # ── Dataset 2: generalization pooled ──
    Xg, yg, wvl_g, _, _ = load_generalization_data(
        str(gen_csv()), band_range=BAND)
    v2g, evrg = pc2_loadings(Xg)
    df2 = extract_extrema(wvl_g, v2g)
    df2.insert(0, "dataset", "Generalization_Pooled")
    print(f"\n[Generalization pooled] PC1={evrg[0]*100:.2f}%  PC2={evrg[1]*100:.2f}%")
    print(df2.to_string(index=False))

    out = BASE_DIR / "results_pc2_extrema.csv"
    pd.concat([df1, df2], ignore_index=True).to_csv(
        out, index=False, float_format="%.4f")
    print(f"\nSaved: {out}")

    # ── Annotated figure (Original Clear PC2) ──
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(wvl, v2, color="#d62728", lw=1.5, label="PC2 loading")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvspan(1340, 1460, alpha=0.10, color="gray", label="Water gap")
    for _, r in df1.iterrows():
        off = 12 if r["type"] == "peak" else -18
        ax.annotate(f"{r['wavelength_nm']:.0f}",
                    (r["wavelength_nm"], r["loading"]),
                    textcoords="offset points", xytext=(0, off),
                    ha="center", fontsize=9)
        ax.plot(r["wavelength_nm"], r["loading"], "k.", ms=6)
    ax.set_xlabel("Wavelength (nm)", fontsize=13)
    ax.set_ylabel("PC2 loading (a.u.)", fontsize=13)
    ax.set_title("PC2 loading vector extrema — Original dataset (Clear)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, ls="--")
    plt.tight_layout()
    fig_path = BASE_DIR / "fig_pc2_annotated.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
