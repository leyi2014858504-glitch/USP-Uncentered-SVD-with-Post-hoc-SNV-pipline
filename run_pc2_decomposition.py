#!/usr/bin/env python3
"""
Decompose PC2 loading: scattering baseline vs water absorption vs polymer C-H.

Question from reviewer: what vibrations do the PC2 extrema correspond to?
Candidate generators (all available within 940-1680 nm):
  (i)  scattering slope   : smooth monotonic lambda-dependence, e.g. Mie
                            lambda^-n  -> captured by poly(log lambda), deg 3
  (ii) water absorption   : literature mu_abs(lambda) for pure liquid water
  (iii) polymer C-H       : 1st overtone 2nu_CH ~1700-1730 nm and nu+delta
                            ~2300 nm lie OUTSIDE/at the edge of the window;
                            only weak 3nu/4nu CH (~1180-1210, ~1430) are inside

Method: least-squares fit of the PC2 loading vector onto the design matrix
[baseline poly, water template], report R^2 and partial contributions, plus
cross-dataset agreement of the extrema positions (if PC2 were polymer
chemistry, two different polymer sets would put extrema at different
wavelengths).
"""
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from run_pc2_band_assignment import (
    datasets, fit_pc, water_template, BAND)

BASE = Path(__file__).parent


def r2(y, yhat):
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    store = {}
    for name, X, y, wvl in datasets():
        V, energy, _ = fit_pc(X)
        v2 = V[1]
        store[name] = (wvl, v2, energy)

        # design matrix
        ln = np.log(wvl / 1000.0)
        B_base = np.column_stack([np.ones_like(ln), ln, ln ** 2, ln ** 3])
        wt = water_template(wvl)
        wt_n = (wt - wt.mean()) / wt.std()

        def fit(D):
            c, *_ = np.linalg.lstsq(D, v2, rcond=None)
            return c, D @ c

        c_b, p_b = fit(B_base)
        c_w, p_w = fit(wt_n.reshape(-1, 1))
        c_f, p_f = fit(np.column_stack([B_base, wt_n]))

        ss_b = ((p_b - v2.mean()) ** 2).sum()
        ss_w = ((p_w - v2.mean()) ** 2).sum()
        ss_f = ((p_f - v2.mean()) ** 2).sum()
        ss_tot = ((v2 - v2.mean()) ** 2).sum()

        print("=" * 88)
        print(f"{name}  (PC2 energy = {energy[1]*100:.2f}%)")
        print(f"  R^2  scattering baseline (cubic in log lambda) : {ss_b/ss_tot:.3f}")
        print(f"  R^2  water absorption template                 : {ss_w/ss_tot:.3f}")
        print(f"  R^2  joint model                               : {ss_f/ss_tot:.3f}")
        # incremental: what does water add on top of the smooth baseline?
        r_base_resid = v2 - p_b
        rr, pp = pearsonr(r_base_resid, wt)
        print(f"  corr(residual-after-baseline, water template)  : {rr:+.3f} (p={pp:.2g})")
        print(f"  -> incremental R^2 from water = "
              f"{(ss_f - ss_b)/ss_tot:.3f}")
        print(f"  water coefficient (a.u. per cm^-1)             : {c_f[-1]:+.5f}")

    # ── cross-dataset agreement of extrema positions ──
    print("\n" + "=" * 88)
    print("Cross-dataset agreement of PC2 shape (different polymer sets):")
    print("  Original  = PET/HDPE/LDPE/PP/EPSF/Weathered")
    print("  General.  = PE/PET/PLA/PP/PVC/SBR")
    a, b = list(store.values())
    # both share the same wavelength grid (940-1680, 741 bands)
    va, vb = a[1], b[1]
    r, p = pearsonr(va, vb)
    print(f"  corr(PC2_original, PC2_generalization) = {r:+.3f}  (p={p:.2g})")
    print(f"  -> {'CONSISTENT: common physical origin (water/scattering)' if r > 0.6 else 'DIVERGENT: dataset-specific'}")

    print("\n  Reference band positions vs observed PC2 extrema:")
    print("    water maxima   :  970, 1200, 1450 nm")
    print("    water windows  : 1070, 1290, 1600 nm")
    print("    C-H 3nu (weak) : ~1180-1210 nm")
    print("    C-H 2nu (STRONG): ~1700-1730 nm  <-- OUTSIDE 940-1680 window")
    print("    C-H nu+delta   : ~2300-2390 nm   <-- OUTSIDE 940-1680 window")


if __name__ == "__main__":
    main()
