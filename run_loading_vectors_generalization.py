#!/usr/bin/env python3
"""
run_loading_vectors_generalization.py
=====================================
Plot PC1 & PC2 loading vectors (uncentered TruncatedSVD) for the two
new datasets:

  1. __Mean_Spectra_Reflectance.csv  (228 samples, 6 polymers, 6 conditions)
  2. ABS_plastic.csv                  (500 ABS samples, no labels)

Both datasets are stored in transposed format (rows = wavelengths,
columns = samples).  This script loads them, fits uncentered SVD on
the full sample set, and plots the first two right-singular vectors
(V[0], V[1]) as loading vectors.

Also reports PC1 energy ratio (variance explained) for diagnostic
comparison with the original plastic dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

BASE_DIR = Path(__file__).parent


def load_transposed_csv(csv_path, band_range=(940, 1680)):
    """Load a transposed CSV (rows=wavelengths, cols=samples).

    Handles two layouts:
      Layout A (e.g. __Mean_Spectra_Reflectance.csv):
        - First column = wavelength value (nm), header = 'wavelength'
        - Remaining columns = sample spectra
      Layout B (e.g. ABS_plastic.csv):
        - First column = wavelength index (0, 1, 2, ...), no header
        - Second column = wavelength value (nm), header = 'Wavelength (nm)'
        - Remaining columns = sample spectra

    Returns
    -------
    X : (n_samples, n_bands) float32
    wavelengths : (n_bands,) float32
    """
    df = pd.read_csv(csv_path, index_col=0)

    # Find the wavelength column: either 'Wavelength (nm)' or 'wavelength'
    wvl_col = None
    for c in df.columns:
        if 'wavelength' in c.lower() or 'wave' in c.lower():
            wvl_col = c
            break

    if wvl_col is None:
        # Layout A: index IS the wavelength
        df = df.reset_index()
        wvl_col = df.columns[0]  # the original index, now a column

    # Coerce to float
    df[wvl_col] = pd.to_numeric(df[wvl_col], errors='coerce')
    df = df.dropna(subset=[wvl_col])

    # Filter rows by band range
    mask = (df[wvl_col] >= band_range[0]) & (df[wvl_col] <= band_range[1])
    df_filtered = df.loc[mask].sort_values(wvl_col)

    wavelengths = df_filtered[wvl_col].values.astype(np.float32)

    # Sample columns = all columns except the wavelength column
    sample_cols = [c for c in df_filtered.columns if c != wvl_col]
    X_T = df_filtered[sample_cols].values
    # X_T shape: (n_bands, n_samples) → transpose
    X = X_T.T.astype(np.float32)

    return X, wavelengths


def plot_loading_vectors(X, wavelengths, title, output_path=None,
                         n_components=20, water_gap=True):
    """Fit uncentered SVD on X and plot PC1/PC2 loading vectors.

    Parameters
    ----------
    X : (n_samples, n_bands)
    wavelengths : (n_bands,)
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)
    V = svd.components_  # shape: (n_components, n_bands)

    # PC1 energy ratio
    explained = svd.explained_variance_ratio_
    pc1_ratio = explained[0] * 100
    pc2_ratio = explained[1] * 100
    pc1_pc2_sum = (explained[0] + explained[1]) * 100

    print(f"  PC1 energy ratio: {pc1_ratio:.2f}%")
    print(f"  PC2 energy ratio: {pc2_ratio:.2f}%")
    print(f"  PC1+PC2:          {pc1_pc2_sum:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(wavelengths, V[0, :], label=f'PC1 Loading ({pc1_ratio:.1f}%)',
            color='black', linewidth=1.8, linestyle='-')
    ax.plot(wavelengths, V[1, :], label=f'PC2 Loading ({pc2_ratio:.1f}%)',
            color='#d62728', linewidth=1.4, linestyle='--')

    if water_gap:
        ax.axvspan(1340, 1460, alpha=0.10, color='gray',
                   label='Water absorption gap')

    ax.set_xlabel('Wavelength (nm)', fontsize=15)
    ax.set_ylabel('Loading Value (Arbitrary Units)', fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)
    return fig, ax, svd


def run():
    print("=" * 80)
    print("Loading Vectors for Generalization Datasets")
    print("(Uncentered TruncatedSVD, k=20, 940-1680 nm)")
    print("=" * 80)

    # ─── Dataset 1: __Mean_Spectra_Reflectance.csv ───
    csv1 = BASE_DIR / "__Mean_Spectra_Reflectance.csv"
    if csv1.exists():
        print(f"\n[1] {csv1.name}")
        X1, wvl1 = load_transposed_csv(csv1, band_range=(940, 1680))
        print(f"  Loaded: {X1.shape[0]} samples × {X1.shape[1]} bands")
        print(f"  Wavelength range: {wvl1[0]:.0f} - {wvl1[-1]:.0f} nm")

        fig, ax, svd1 = plot_loading_vectors(
            X1, wvl1,
            title="Loading Vectors — Generalization Dataset (228 samples, 6 polymers)",
            output_path=BASE_DIR / "fig_loading_vectors_generalization.png",
        )
    else:
        print(f"\n[1] {csv1.name} — NOT FOUND")

    # ─── Dataset 2: ABS_plastic.csv ───
    csv2 = BASE_DIR / "ABS_plastic.csv"
    if csv2.exists():
        print(f"\n[2] {csv2.name}")
        X2, wvl2 = load_transposed_csv(csv2, band_range=(940, 1680))
        print(f"  Loaded: {X2.shape[0]} samples × {X2.shape[1]} bands")
        print(f"  Wavelength range: {wvl2[0]:.0f} - {wvl2[-1]:.0f} nm")

        fig, ax, svd2 = plot_loading_vectors(
            X2, wvl2,
            title="Loading Vectors — ABS Plastic Dataset (500 samples)",
            output_path=BASE_DIR / "fig_loading_vectors_abs.png",
        )
    else:
        print(f"\n[2] {csv2.name} — NOT FOUND")

    print("\nDone!")


if __name__ == "__main__":
    run()
