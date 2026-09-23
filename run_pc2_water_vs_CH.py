#!/usr/bin/env python3
"""
run_pc2_water_vs_CH.py
======================
Discriminating test for the origin of the PC2 loading extrema.

Both plastic datasets are (a) immersed in water and (b) C-H-rich polymers,
so their near-identical PC2 shape (r=0.962) does NOT by itself separate
"water absorption" from "polymer C-H overtones" — the two bands overlap:

    1215 nm : water O-H (nu+delta)   ~  C-H 3nu          (coincident!)
    1440 nm : water O-H 2nu          ~  C-H 2delta       (coincident!)
    1680 nm : water window rising    ~  C-H 2nu onset    (coincident!)

Control dataset: textile (DRY fabric, no water background, still C-H-rich
cellulose / polyester / lyocell).

Logic:
  - If PC2 extrema are polymer C-H -> textile PC2 should show the SAME
    peak positions (1215, 1440) and correlate with plastic PC2.
  - If PC2 extrema are water -> textile PC2 should NOT track the water
    absorption template, and its shape should diverge from plastic PC2.

Also compares PC2 across the three water conditions of the original
dataset (same polymers, different water turbidity).

Outputs: results_pc2_water_vs_CH.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD

from run_log_preprocessing import load_data
from run_generalization_experiment import load_generalization_data
from run_textile_bandshift import load_textile_data
from run_pc2_band_assignment import water_template, REFERENCE_BANDS

BASE = Path(__file__).parent
D = BASE / "1"
BAND = (940, 1680)
N_DIM = 20

# diagnostic band positions (nm): water-specific vs C-H-specific
PROBE = [
    (1070, "water window (mu~0.05)"),
    (1215, "water nu+delta  OR  C-H 3nu"),
    (1285, "water window"),
    (1445, "water 2nu  OR  C-H 2delta"),
    (1600, "water window / C-H region"),
]


def pc2(X):
    Xd = X.astype(np.float64)
    k = min(N_DIM, Xd.shape[0] - 1, Xd.shape[1])
    svd = TruncatedSVD(n_components=k, random_state=42).fit(Xd)
    S2 = svd.singular_values_ ** 2
    unc = S2 / (Xd ** 2).sum()
    return svd.components_[1], unc[1], svd.explained_variance_ratio_[1]


def sample_at(wvl, v, targets):
    idx = [int(np.argmin(np.abs(wvl - t))) for t in targets]
    return v[idx]


def main():
    sets = {}

    X, y = load_data(str(D / "spectradictionary.csv"), water_filter=0)
    w = np.arange(BAND[0], BAND[1] + 1, dtype=float)[: X.shape[1]]
    sets["plastic_Clear"] = (w,) + pc2(X)

    X, y = load_data(str(D / "spectradictionary.csv"), water_filter=1)
    sets["plastic_Turbid"] = (w,) + pc2(X)

    X, y = load_data(str(D / "spectradictionary.csv"), water_filter=2)
    sets["plastic_Foamy"] = (w,) + pc2(X)

    Xg, yg, wvg, _, _ = load_generalization_data(
        str(BASE / "__Mean_Spectra_Reflectance.csv"), band_range=BAND)
    sets["plastic_Generalization"] = (wvg.astype(float),) + pc2(Xg)

    # DIW = deionised water, lowest turbidity -> strongest water transmission
    Xd, yd, wvd, _, _ = load_generalization_data(
        str(BASE / "__Mean_Spectra_Reflectance.csv"),
        band_range=BAND, condition_filter="DIW")
    sets["plastic_Gen_DIW"] = (wvd.astype(float),) + pc2(Xd)

    # CONTROL: dry textile, no water background
    sp, gt = D / "swir_mean_spectra.csv", D / "ground_truth_final.csv"
    if sp.exists() and gt.exists():
        Xt, yt, wvt = load_textile_data(sp, gt, band_range=(1000, 1680),
                                        min_dominant_frac=0.0)
        sets["textile_DRY"] = (wvt.astype(float),) + pc2(Xt)

    rows = []
    print("=" * 96)
    print("PC2 loading at diagnostic bands (normalised to [0,1] over each vector)")
    print("=" * 96)
    hdr = f"{'dataset':<24}" + "".join(f"{t:>8}" for t, _ in PROBE) + \
          f"{'r(PC2,H2O)':>12}{'PC2%unc':>9}{'PC2%sk':>8}"
    print(hdr)
    print("-" * 96)
    for name, (wv, v2, unc, sk) in sets.items():
        vt = water_template(wv)
        r_w, _ = pearsonr(v2, vt)
        vals = sample_at(wv, v2, [t for t, _ in PROBE])
        rng = v2.max() - v2.min()
        print(f"{name:<24}" + "".join(f"{(x - v2.min())/rng:>8.3f}" for x in vals)
              + f"{r_w:>+12.3f}{unc*100:>9.2f}{sk*100:>8.2f}")
        rows.append({"dataset": name, "pc2_energy_uncentered_pct": unc * 100,
                     "pc2_energy_sklearn_pct": sk * 100,
                     "corr_pc2_water_template": r_w,
                     **{f"norm_load_{t}nm": (v - v2.min()) / rng
                        for t, v in zip([p[0] for p in PROBE], vals)}})

    # ── shape correlations against the dry-textile control ──
    print("\n" + "=" * 96)
    print("PC2 shape correlation matrix (interpolated to common 1000-1680 grid)")
    print("=" * 96)
    grid = np.arange(1000, 1681, dtype=float)
    interp = {n: np.interp(grid, wv, v2) for n, (wv, v2, _, _) in sets.items()}
    names = list(sets)
    print(f"{'':<24}" + "".join(f"{n[:9]:>10}" for n in names))
    for a in names:
        line = f"{a:<24}"
        for b in names:
            r, _ = pearsonr(interp[a], interp[b])
            line += f"{r:>10.3f}"
        print(line)

    if "textile_DRY" in sets:
        print("\nKEY: row/col 'textile_DRY' = dry control (no water).")
        for n in names:
            if n == "textile_DRY":
                continue
            r, p = pearsonr(interp[n], interp["textile_DRY"])
            print(f"  {n:<24} vs textile_DRY : r = {r:+.3f} (p={p:.2g})")

    pd.DataFrame(rows).to_csv(BASE / "results_pc2_water_vs_CH.csv",
                              index=False, float_format="%.6f")
    print("\nSaved: results_pc2_water_vs_CH.csv")


if __name__ == "__main__":
    main()
