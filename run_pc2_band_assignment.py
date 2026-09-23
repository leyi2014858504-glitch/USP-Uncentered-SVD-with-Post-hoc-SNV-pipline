#!/usr/bin/env python3
"""
run_pc2_band_assignment.py
==========================
Reviewer request: do not merely say PC2 "has peaks/valleys" or "coincides
with the C-H region" — quantify the extrema (exact wavelength + loading
value) and discuss which molecular vibrations they may correspond to.

Outputs, for both datasets (uncentered TruncatedSVD, k=20, 940-1680 nm):
  1. PC2 extrema table  (peak/trough wavelength, loading, prominence)
  2. PC2 loading sampled at literature reference band positions
  3. Correlation of PC2 with a pure-water absorption template
     (alternation at water maxima/minima is the diagnostic signature)
  4. Corrected PC1/PC2 energy (true uncentered sigma^2 / sum X^2)

Note on convention: sklearn's explained_variance_ratio_ divides by the
CENTERED total sum of squares even for TruncatedSVD, which is an invalid
ratio for uncentered decompositions (it can exceed 100%). We therefore
report the true uncentered energy fraction sigma_i^2 / sum(X^2).

Outputs:
  results_pc2_extrema.csv
  results_pc2_band_assignment.csv
  fig_pc2_annotated.png
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD

from run_log_preprocessing import load_data
from run_generalization_experiment import load_generalization_data

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"
N_DIM = 20
BAND = (940, 1680)


# ─────────────────── Literature reference bands ───────────────────
# (position_nm, assignment, kind)  kind: 'max' = absorption band,
#                                   'min' = transmission window
REFERENCE_BANDS = [
    (970,  "H2O O-H nu1+nu3 combination", "max"),
    (1070, "H2O transmission window", "min"),
    (1180, "C-H 3nu / nu+2delta (polymer, weak)", "max"),
    (1210, "H2O O-H nu+delta combination", "max"),
    (1290, "H2O transmission window", "min"),
    (1410, "C-H combination (PVC/PET, weak)", "max"),
    (1450, "H2O O-H 2nu1 first overtone", "max"),
    (1510, "N-H 1st overtone (amide, PA only)", "max"),
    (1600, "H2O transmission window", "min"),
    (1660, "aromatic C-H 1st overtone (PET)", "max"),
    (1680, "C-H 1st overtone onset (PE/PP/PET)", "max"),
]

# Pure-water absorption coefficient anchors (cm^-1), literature values.
# Used to build an interpolated template over 940-1680 nm.
WATER_ANCHORS = [
    (940, 0.24), (965, 0.40), (1000, 0.20), (1050, 0.07), (1070, 0.05),
    (1100, 0.09), (1150, 0.22), (1200, 0.45), (1250, 0.28), (1300, 0.11),
    (1350, 0.16), (1400, 0.45), (1440, 1.00), (1470, 0.78), (1500, 0.48),
    (1550, 0.24), (1600, 0.16), (1650, 0.26), (1680, 0.36),
]


def water_template(wvl):
    aw = np.array([a[0] for a in WATER_ANCHORS], dtype=float)
    ac = np.array([a[1] for a in WATER_ANCHORS], dtype=float)
    return np.interp(wvl, aw, ac)


# ─────────────────── Core computation ───────────────────

def fit_pc(X):
    """Return (V, true uncentered energy fractions, evr_sklearn)."""
    Xd = X.astype(np.float64)
    svd = TruncatedSVD(n_components=N_DIM, random_state=42).fit(Xd)
    S2 = svd.singular_values_ ** 2
    energy = S2 / (Xd ** 2).sum()          # true uncentered fraction
    return svd.components_, energy, svd.explained_variance_ratio_


def extract_extrema(wvl, v2, min_prom_frac=0.08):
    """Peaks/troughs of PC2 with prominence >= 8% of the loading range."""
    rng = v2.max() - v2.min()
    thr = min_prom_frac * rng
    pk, pk_p = find_peaks(v2, prominence=thr)
    tr, tr_p = find_peaks(-v2, prominence=thr)
    rows = [{"wavelength_nm": float(wvl[i]), "loading": float(v2[i]),
             "type": "peak", "prominence": float(p),
             "norm_loading": float((v2[i] - v2.min()) / rng)}
            for i, p in zip(pk, pk_p["prominences"])]
    rows += [{"wavelength_nm": float(wvl[i]), "loading": float(v2[i]),
              "type": "trough", "prominence": float(p),
              "norm_loading": float((v2[i] - v2.min()) / rng)}
             for i, p in zip(tr, tr_p["prominences"])]
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("prominence", ascending=False).reset_index(drop=True)
    return df


def loading_at(wvl, v2, targets):
    idx = [int(np.argmin(np.abs(wvl - t))) for t in targets]
    return wvl[idx], v2[idx]


# ─────────────────── Data sources ───────────────────

def datasets():
    # Original: clear-water river plastics
    p_orig = DATA_DIR / "spectradictionary.csv"
    if not p_orig.exists():
        p_orig = BASE_DIR / "spectradictionary.csv"
    X, y = load_data(str(p_orig), water_filter=0)
    wvl = np.arange(BAND[0], BAND[1] + 1, dtype=float)[: X.shape[1]]
    yield ("Original_Clear", X, y, wvl)

    # Generalization: 6 polymers, pooled across water conditions
    p_gen = DATA_DIR / "__Mean_Spectra_Reflectance.csv"
    if not p_gen.exists():
        p_gen = BASE_DIR / "__Mean_Spectra_Reflectance.csv"
    Xg, yg, wvlg, _, _ = load_generalization_data(
        str(p_gen), band_range=BAND)
    yield ("Generalization_Pooled", Xg, yg, wvlg.astype(float))


# ─────────────────── Main ───────────────────

def main():
    datasets_list = list(datasets())
    ext_rows, asg_rows = [], []
    fig, axes = plt.subplots(len(datasets_list), 1,
                             figsize=(12, 4.6 * len(datasets_list)),
                             squeeze=False)

    for k, (name, X, y, wvl) in enumerate(datasets_list):
        V, energy, evr_sk = fit_pc(X)
        v2 = V[1]
        rng = v2.max() - v2.min()

        print("=" * 92)
        print(f"{name}: n={len(y)}  bands={X.shape[1]}  "
              f"({wvl[0]:.0f}-{wvl[-1]:.0f} nm)")
        print(f"  true uncentered energy:  PC1 = {energy[0]*100:.2f}%   "
              f"PC2 = {energy[1]*100:.2f}%")
        print(f"  (sklearn evr, wrong centered denom: "
              f"PC1 = {evr_sk[0]*100:.2f}%  PC2 = {evr_sk[1]*100:.2f}%)")

        # ── extrema ──
        ex = extract_extrema(wvl, v2)
        ex.insert(0, "dataset", name)
        ext_rows.append(ex)
        print(f"\n  PC2 extrema (prominence >= 8% of range):")
        print(f"    {'type':<8}{'lambda_nm':>11}{'loading':>11}"
              f"{'norm':>8}{'prominence':>13}")
        for _, r in ex.iterrows():
            print(f"    {r['type']:<8}{r['wavelength_nm']:>11.0f}"
                  f"{r['loading']:>11.5f}{r['norm_loading']:>8.3f}"
                  f"{r['prominence']:>13.5f}")

        # ── loading at reference bands ──
        tgt = [t for t, _, _ in REFERENCE_BANDS]
        la, va = loading_at(wvl, v2, tgt)
        print(f"\n  PC2 loading at literature reference positions:")
        print(f"    {'ref_nm':>7}{'act_nm':>8}{'loading':>10}{'norm':>7}"
              f"  {'kind':<5} assignment")
        for (t, desc, kind), a, v in zip(REFERENCE_BANDS, la, va):
            nv = (v - v2.min()) / rng
            print(f"    {t:>7}{a:>8.0f}{v:>10.5f}{nv:>7.3f}  {kind:<5} {desc}")
            asg_rows.append({
                "dataset": name, "reference_nm": t, "actual_nm": float(a),
                "kind": kind, "assignment": desc,
                "pc2_loading": float(v), "pc2_norm": float(nv),
            })

        # ── water-template correlation ──
        wt = water_template(wvl)
        wtn = (wt - wt.min()) / (wt.max() - wt.min())
        v2n = (v2 - v2.min()) / rng
        r_w, p_w = pearsonr(v2n, wtn)
        print(f"\n  corr(PC2_norm, water-absorption template) = "
              f"{r_w:+.3f}  (p={p_w:.4f})")
        # alternation test: PC2 sign at water absorption maxima vs windows
        w_max = [v for (t, d, kind), a, v in zip(REFERENCE_BANDS, la, va)
                 if kind == "max" and "H2O" in d]
        w_min = [v for (t, d, kind), a, v in zip(REFERENCE_BANDS, la, va)
                 if kind == "min" and "H2O" in d]
        p_max = [v for (t, d, kind), a, v in zip(REFERENCE_BANDS, la, va)
                 if kind == "max" and "H2O" not in d]
        print(f"  mean PC2 at H2O maxima   (n={len(w_max)}): "
              f"{np.mean(w_max):+.5f}")
        print(f"  mean PC2 at H2O windows  (n={len(w_min)}): "
              f"{np.mean(w_min):+.5f}")
        print(f"  mean PC2 at C-H/N-H bands(n={len(p_max)}): "
              f"{np.mean(p_max):+.5f}")
        print(f"  mean|PC2| by region:  <1100nm={np.abs(v2[wvl < 1100]).mean():.5f}"
              f"   1300-1600nm={np.abs(v2[(wvl > 1300) & (wvl < 1600)]).mean():.5f}"
              f"   >1600nm={np.abs(v2[wvl > 1600]).mean():.5f}")

        # ── annotated subplot ──
        ax = axes[k, 0]
        ax.plot(wvl, v2, color="#d62728", lw=1.5,
                label=f"PC2 loading ({energy[1]*100:.2f}% energy)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvspan(1340, 1420, alpha=0.08, color="blue", label="H2O window")
        for _, r in ex.iterrows():
            ax.plot(r["wavelength_nm"], r["loading"], "k.", ms=7)
            off = 13 if r["type"] == "peak" else -19
            ax.annotate(f"{r['wavelength_nm']:.0f} nm",
                        (r["wavelength_nm"], r["loading"]),
                        textcoords="offset points", xytext=(0, off),
                        ha="center", fontsize=9)
        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel("PC2 loading (a.u.)", fontsize=12)
        ax.set_title(f"PC2 loading vector — {name}", fontsize=13, loc="left")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(alpha=0.3, ls="--")
        ax2 = ax.twinx()
        ax2.plot(wvl, water_template(wvl), color="steelblue", lw=1.0,
                 ls=":", alpha=0.8, label="H2O absorption coeff.")
        ax2.set_ylabel("H2O abs. coeff. (cm$^{-1}$)", fontsize=10,
                       color="steelblue")
        ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_fig = BASE_DIR / "fig_pc2_annotated.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_ext = pd.concat(ext_rows, ignore_index=True)
    df_ext.to_csv(BASE_DIR / "results_pc2_extrema.csv",
                  index=False, float_format="%.6f")
    df_asg = pd.DataFrame(asg_rows)
    df_asg.to_csv(BASE_DIR / "results_pc2_band_assignment.csv",
                  index=False, float_format="%.6f")
    print(f"\nSaved: results_pc2_extrema.csv  ({len(df_ext)} rows)")
    print(f"Saved: results_pc2_band_assignment.csv  ({len(df_asg)} rows)")
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
