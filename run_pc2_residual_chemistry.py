#!/usr/bin/env python3
"""
run_pc2_residual_chemistry.py
=============================
Decisive test: after removing the SMOOTH scattering baseline from PC2,
does the remaining BAND STRUCTURE reflect polymer chemistry (C-H) or
water absorption?

Rationale
---------
A smooth monotonic baseline (cubic in log lambda) is present in ANY
reflectance dataset and carries no band-position information. The
scientifically meaningful part of a loading vector is the residual
band structure. We compare that residual across:
  - plastics in water (Clear / Turbid / Foamy / Generalization)
  - dry textile (no water background, but C-H-rich cellulose/PET/lyocell)

If residual band structure is shared between wet plastic and DRY textile,
it cannot be water -> it is C-H (or a generic instrument/spectral shape).
If it differs, the water assignment stands.

SVD sign is arbitrary -> all vectors are sign-aligned to plastic_Clear
before correlation.
"""
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD

from run_log_preprocessing import load_data
from run_generalization_experiment import load_generalization_data
from run_textile_bandshift import load_textile_data
from run_pc2_band_assignment import water_template

BASE = Path(__file__).parent
D = BASE / "1"
BAND = (940, 1680)
N_DIM = 20


def pc2(X):
    Xd = X.astype(np.float64)
    k = min(N_DIM, Xd.shape[0] - 1, Xd.shape[1])
    svd = TruncatedSVD(n_components=k, random_state=42).fit(Xd)
    return svd.components_[1]


def baseline(v, wvl, deg=3):
    ln = np.log(wvl / 1000.0)
    A = np.vander(ln, deg + 1)
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return A @ c


def load_all():
    out = {}
    w = np.arange(BAND[0], BAND[1] + 1, dtype=float)
    for wl, wf in [("Clear", 0), ("Turbid", 1), ("Foamy", 2)]:
        X, y = load_data(str(D / "spectradictionary.csv"), water_filter=wf)
        out[f"plastic_{wl}"] = (w[: X.shape[1]], pc2(X))
    Xg, _, wvg, _, _ = load_generalization_data(
        str(BASE / "__Mean_Spectra_Reflectance.csv"), band_range=BAND)
    out["plastic_Gen"] = (wvg.astype(float), pc2(Xg))
    for cond in ["DIW", "TG", "Bu5"]:
        Xc, _, wvc, _, _ = load_generalization_data(
            str(BASE / "__Mean_Spectra_Reflectance.csv"),
            band_range=BAND, condition_filter=cond)
        out[f"plastic_{cond}"] = (wvc.astype(float), pc2(Xc))
    Xt, _, wvt = load_textile_data(D / "swir_mean_spectra.csv",
                                   D / "ground_truth_final.csv",
                                   band_range=(1000, 1680),
                                   min_dominant_frac=0.0)
    out["textile_DRY"] = (wvt.astype(float), pc2(Xt))
    return out


def main():
    data = load_all()
    grid = np.arange(1000, 1681, dtype=float)

    ref = None
    proc = {}
    for name, (wv, v2) in data.items():
        v = np.interp(grid, wv, v2)
        r = v - baseline(v, grid)          # residual band structure
        if ref is None:
            ref = r                        # sign reference = plastic_Clear
        if np.corrcoef(r, ref)[0, 1] < 0:
            v, r = -v, -r
        proc[name] = (v, r)

    print("=" * 92)
    print("Residual band structure (PC2 minus cubic-in-log-lambda baseline)")
    print("=" * 92)
    print(f"{'dataset':<18}{'baseline R2':>13}{'resid var%':>12}"
          f"{'corr(resid,H2O)':>17}{'corr(full,H2O)':>16}")
    print("-" * 92)
    wt = np.interp(grid, grid, water_template(grid))
    for name, (v, r) in proc.items():
        b = v - r
        rb = 1 - ((v - b) ** 2).sum() / ((v - v.mean()) ** 2).sum()
        rv = (r ** 2).sum() / ((v - v.mean()) ** 2).sum()
        rr, pr = pearsonr(r, wt)
        rf, pf = pearsonr(v, wt)
        print(f"{name:<18}{rb:>13.3f}{rv:>11.1%}{rr:>+12.3f} (p={pr:5.1g})"
              f"{rf:>+10.3f} (p={pf:5.1g})")

    print("\n" + "=" * 92)
    print("RESIDUAL cross-correlation  (chemistry test)")
    print("=" * 92)
    names = list(proc)
    print(f"{'':<18}" + "".join(f"{n.replace('plastic_','').replace('textile_','DRY_')[:8]:>10}"
                                for n in names))
    for a in names:
        line = f"{a:<18}"
        for b in names:
            r, _ = pearsonr(proc[a][1], proc[b][1])
            line += f"{r:>10.3f}"
        print(line)

    print("\nKEY COMPARISON — dry textile vs wet plastics (residual only):")
    for n in names:
        if n == "textile_DRY":
            continue
        r, p = pearsonr(proc["textile_DRY"][1], proc[n][1])
        tag = "SHARED (=> C-H / chemistry)" if (r > 0.4 and p < 0.01) else \
              "NOT shared (=> water-specific)"
        print(f"  textile_DRY vs {n:<16} r = {r:+.3f} (p={p:.2g})   {tag}")

    print("\nFULL-VECTOR cross-correlation (for reference, baseline included):")
    for n in names:
        if n == "textile_DRY":
            continue
        r, p = pearsonr(proc["textile_DRY"][0], proc[n][0])
        print(f"  textile_DRY vs {n:<16} r = {r:+.3f} (p={p:.2g})")

    # ── where do the residuals peak?  report wavelengths ──
    from scipy.signal import find_peaks
    print("\n" + "=" * 92)
    print("Dominant residual band positions (|prominence| ranked)")
    print("=" * 92)
    for n in ["plastic_Clear", "plastic_Gen", "textile_DRY"]:
        r = proc[n][1]
        rng = r.max() - r.min()
        pk, pp = find_peaks(r, prominence=0.15 * rng)
        tr, tp = find_peaks(-r, prominence=0.15 * rng)
        items = sorted([(grid[i], "peak", v) for i, v in zip(pk, pp["prominences"])] +
                       [(grid[i], "trough", v) for i, v in zip(tr, tp["prominences"])],
                       key=lambda x: -x[2])[:5]
        print(f"\n  {n}:")
        for lam, kind, prom in items:
            print(f"    {lam:7.0f} nm  {kind:<7} prominence={prom:.4f}")


if __name__ == "__main__":
    main()
