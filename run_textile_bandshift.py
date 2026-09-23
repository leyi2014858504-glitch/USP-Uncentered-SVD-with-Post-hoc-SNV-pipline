#!/usr/bin/env python3
"""
run_textile_bandshift.py
========================
BandShift robustness experiment on the OpenTextile SWIR dataset.

APPENDIX-ONLY: Single fixed 4:1 split (seed=42), no statistical tests.
  - Same 5 fair methods (USP / SNV / SG1 / EMSC / Detrend)
  - 2 classifiers (LinearSVM / RBF-SVM) — RF dropped
  - Single stratified 4:1 split (seed=42)
  - Evaluate the same trained model on:
      (a) normal test set
      (b) BandShift-perturbed test set (np.roll(X, shift=2, axis=1))

Setup:
  - Textile dataset (57 samples after filtering, 3 fiber classes:
    Cotton / Polyester / Lyocell-Modacrylic)
  - Wavelength range: 1000-1680 nm (overlap with plastic dataset)

Outputs:
  results_textile_bandshift_full.csv  — summary table
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"
N_DIM = 20
SPLIT_SEED = 42  # Single fixed split for appendix


# ───────────────────────── Data ─────────────────────────

FIBER_NAMES = {
    0: "Cotton", 1: "Polyester", 2: "Wool",
    3: "Polyamide", 4: "Viscose", 5: "Lyocell/Modacrylic"
}


def load_textile_data(spectra_path, gt_path, band_range=(1000, 1680),
                      min_dominant_frac=0.0, exclude_unknown=True,
                      exclude_outlier=True):
    """Same loader as run_textile_experiment.py."""
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

    return X, y, wavelengths


# ───────────────────────── Preprocessing ─────────────────────────

def snv(X, eps=1e-8):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def sg_first_derivative(X, window=15, polyorder=2):
    # window must be odd and <= signal length
    n_bands = X.shape[1]
    win = min(window, n_bands if n_bands % 2 == 1 else n_bands - 1)
    if win % 2 == 0:
        win -= 1
    win = max(win, 5)
    return savgol_filter(X, window_length=win, polyorder=polyorder,
                         deriv=1, delta=1.0, axis=1).astype(np.float32)


def emsc(X_train, X_test=None, poly_order=2):
    n_bands = X_train.shape[1]
    ref = X_train.mean(axis=0)
    w = np.linspace(-1, 1, n_bands)

    cols = [np.ones(n_bands), ref]
    for d in range(1, poly_order + 1):
        cols.append(w ** d)
    D = np.column_stack(cols)

    def _correct(X):
        Xc = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            coefs, *_ = np.linalg.lstsq(D, X[i], rcond=None)
            b_ref = coefs[1]
            baseline = coefs[0].copy()
            for d in range(1, poly_order + 1):
                baseline = baseline + coefs[1 + d] * w ** d
            if abs(b_ref) < 1e-8:
                b_ref = 1e-8
            Xc[i] = (X[i] - baseline) / b_ref
        return Xc.astype(np.float32)

    Xtr_c = _correct(X_train)
    Xte_c = _correct(X_test) if X_test is not None else None
    return Xtr_c, Xte_c


def detrend_only(X_train, X_test=None, poly_order=2):
    n_bands = X_train.shape[1]
    w = np.linspace(-1, 1, n_bands)
    A = np.vander(w, poly_order + 1, increasing=True)

    def _detrend(X):
        Xd = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            coefs, *_ = np.linalg.lstsq(A, X[i], rcond=None)
            trend = A @ coefs
            Xd[i] = X[i] - trend
        return Xd.astype(np.float32)

    Xtr_d = _detrend(X_train)
    Xte_d = _detrend(X_test) if X_test is not None else None
    return Xtr_d, Xte_d


PREPROCESSORS = {
    "none":     lambda Xtr, Xte: (Xtr, Xte),
    "SNV":      lambda Xtr, Xte: (snv(Xtr), snv(Xte)),
    "SG1":      lambda Xtr, Xte: (sg_first_derivative(Xtr),
                                  sg_first_derivative(Xte)),
    "EMSC":     lambda Xtr, Xte: emsc(Xtr, Xte),
    "Detrend":  lambda Xtr, Xte: detrend_only(Xtr, Xte),
}


# ───────────────────────── Classifiers ─────────────────────────

def make_clf(clf_type='svm'):
    if clf_type == 'rbf_svm':
        return SVC(kernel='rbf', C=1.0, gamma='scale',
                   class_weight='balanced', random_state=42)
    else:
        return SVC(kernel='linear', C=1.0, class_weight='balanced',
                  random_state=42)


CLASSIFIERS = ["svm", "rbf_svm"]  # RF dropped


# ───────────────────────── Methods (fair comparison) ─────────────────────────

METHODS = [
    ("USP",      "none",     "A_postSNV"),     # Proposed
    ("SNV",      "SNV",      "A2_noPostSNV"),  # Traditional baseline
    ("SG1",      "SG1",      "A2_noPostSNV"),
    ("EMSC",     "EMSC",     "A2_noPostSNV"),
    ("Detrend",  "Detrend",  "A2_noPostSNV"),
]


# ───────────────────────── Pipelines ─────────────────────────

def pipeline_post_snv(Xtr_pre, Xte_pre, ytr, clf_type):
    """USP: TruncSVD(k=20) → SNV in PC space → StandardScaler → clf."""
    n_comp = max(1, min(N_DIM, Xtr_pre.shape[0] - 1, Xtr_pre.shape[1]))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Ztr = svd.fit_transform(Xtr_pre)
    Zte = svd.transform(Xte_pre)
    Ztr = snv(Ztr)
    Zte = snv(Zte)
    sc = StandardScaler().fit(Ztr)
    clf = make_clf(clf_type)
    clf.fit(sc.transform(Ztr), ytr)
    y_pred = clf.predict(sc.transform(Zte))
    return y_pred, svd, clf, sc


def pipeline_pre_snv(Xtr_pre, Xte_pre, ytr, clf_type):
    """Traditional: preprocessed → Centered TruncSVD → StandardScaler → clf."""
    mu = Xtr_pre.mean(axis=0)
    n_comp = max(1, min(N_DIM, Xtr_pre.shape[0] - 1, Xtr_pre.shape[1]))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Ztr = svd.fit_transform(Xtr_pre - mu)
    Zte = svd.transform(Xte_pre - mu)
    sc = StandardScaler().fit(Ztr)
    clf = make_clf(clf_type)
    clf.fit(sc.transform(Ztr), ytr)
    y_pred = clf.predict(sc.transform(Zte))
    return y_pred, svd, clf, sc, mu


def apply_transform_bs(Xte_pre_bs, svd, sc, variant, mu=None):
    """Apply trained pipeline to (possibly perturbed) test set."""
    if variant == "A_postSNV":
        Zte = svd.transform(Xte_pre_bs)
        Zte = snv(Zte)
        return sc.transform(Zte)
    else:
        Zte = svd.transform(Xte_pre_bs - mu)
        return sc.transform(Zte)


# ───────────────────────── Evaluation ─────────────────────────

def evaluate(y_true, y_pred):
    return {
        'OA':       float(accuracy_score(y_true, y_pred)),
        'macro_F1': float(f1_score(y_true, y_pred, average='macro',
                                    zero_division=0)),
        'kappa':    float(cohen_kappa_score(y_true, y_pred)),
    }


def bandshift(X, shift=2):
    return np.roll(X, shift=shift, axis=1).astype(np.float32)


# ───────────────────────── Stats ─────────────────────────
# (Removed: RM-ANOVA and Bonferroni — not needed for single-split appendix)


# ───────────────────────── Main ─────────────────────────

def run():
    spectra_csv = DATA_DIR / "swir_mean_spectra.csv"
    gt_csv = DATA_DIR / "ground_truth_final.csv"

    if not spectra_csv.exists() or not gt_csv.exists():
        print("ERROR: textile data not found in '1/' directory.")
        print("Download from https://zenodo.org/records/18269172")
        return

    print("=" * 100)
    print("Textile BandShift Robustness Experiment (Appendix — single fixed split)")
    print("=" * 100)

    X, y, wvl = load_textile_data(spectra_csv, gt_csv,
                                   band_range=(1000, 1680),
                                   min_dominant_frac=0.0)

    print(f"\nDataset: {len(y)} samples | {X.shape[1]} bands "
          f"({wvl[0]:.0f}-{wvl[-1]:.0f} nm) | {len(np.unique(y))} classes")
    for cls in np.unique(y):
        print(f"  {FIBER_NAMES.get(int(cls), cls)}: {(y == cls).sum()} samples")

    # ─── Single fixed 4:1 stratified split ───
    print(f"\nSingle stratified 4:1 split (seed={SPLIT_SEED})")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    print(f"  Train: {len(ytr)} | Test: {len(yte)}")

    Xte_bs = bandshift(Xte, shift=2)

    records = []
    for method_name, pp_name, variant in METHODS:
        pp_fn = PREPROCESSORS[pp_name]
        Xtr_pp, Xte_pp = pp_fn(Xtr, Xte)
        Xte_pp_bs = pp_fn(Xtr, Xte_bs)[1]

        for clf_type in CLASSIFIERS:
            # Normal test
            if variant == "A_postSNV":
                y_pred, svd, clf, sc = pipeline_post_snv(
                    Xtr_pp, Xte_pp, ytr, clf_type)
                mu = None
            else:
                y_pred, svd, clf, sc, mu = pipeline_pre_snv(
                    Xtr_pp, Xte_pp, ytr, clf_type)
            metrics = evaluate(yte, y_pred)
            records.append({
                'condition': 'Normal',
                'method': method_name,
                'preprocess': pp_name, 'variant': variant,
                'classifier': clf_type, **metrics,
            })

            # BandShift test (same trained model)
            Xte_final_bs = apply_transform_bs(Xte_pp_bs, svd, sc, variant, mu=mu)
            y_pred_bs = clf.predict(Xte_final_bs)
            metrics_bs = evaluate(yte, y_pred_bs)
            records.append({
                'condition': 'BandShift',
                'method': method_name,
                'preprocess': pp_name, 'variant': variant,
                'classifier': clf_type, **metrics_bs,
            })

    # ─── Save summary ───
    df = pd.DataFrame(records)
    out_full = BASE_DIR / "results_textile_bandshift_full.csv"
    df.to_csv(out_full, index=False, float_format="%.6f")
    print(f"\nSaved: {out_full}")

    # ─── Print results ───
    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}

    print("\n" + "=" * 100)
    print("Textile BandShift Results (single fixed split, appendix)")
    print("=" * 100)
    print(f"{'Method':<12} {'Classifier':<14} {'OA(Normal)':>12} "
          f"{'OA(BandShift)':>15} {'ΔOA(pp)':>10} {'Macro-F1(N)':>12} "
          f"{'Kappa(N)':>10}")
    print("-" * 85)
    for method_name, _, _ in METHODS:
        for clf in CLASSIFIERS:
            rn = df[(df['condition'] == 'Normal') &
                    (df['method'] == method_name) &
                    (df['classifier'] == clf)].iloc[0]
            rb = df[(df['condition'] == 'BandShift') &
                    (df['method'] == method_name) &
                    (df['classifier'] == clf)].iloc[0]
            delta = (rb['OA'] - rn['OA']) * 100
            print(f"{method_name:<12} {clf_labels[clf]:<14} "
                  f"{rn['OA']*100:>11.2f}% {rb['OA']*100:>14.2f}% "
                  f"{delta:>9.2f} {rn['macro_F1']*100:>11.2f}% "
                  f"{rn['kappa']:>10.4f}")
        print()


if __name__ == "__main__":
    run()
