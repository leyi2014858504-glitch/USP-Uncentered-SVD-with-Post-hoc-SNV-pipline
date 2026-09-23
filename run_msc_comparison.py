#!/usr/bin/env python3
"""
run_msc_comparison.py
=====================
Compare three MSC reference spectrum selection strategies for NIR plastic
classification: mean spectrum (standard), v1 from uncentered SVD (novel),
and median spectrum (additional control).

All MSC variants are followed by:
  Centered TruncatedSVD (k=20) → StandardScaler → Linear SVM

Baselines:
  A  (Proposed USP):   Uncentered TruncSVD → SNV in PC space → StandardScaler → SVM
  A2 (Conventional):   SNV pre-processing → Centered TruncSVD → StandardScaler → SVM

25 random repeats (seeds 42–66), stratified 4:1 train/test split.
Report OA, Macro-F1, Cohen's Kappa (mean ± std).
If A3_v1 improves over A3_mean, also test on BandShift perturbation.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score)
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from usp_pipeline import load_river_csv, MATERIAL_NAMES

N_DIM = 20
SEED_START = 42
N_REPEATS = 25


def snv(X, eps=1e-8):
    """Standard Normal Variate (row-wise z-score)."""
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class MSCMean:
    """MSC using the training-set mean spectrum as reference."""
    def __init__(self):
        self.ref_ = None

    def fit(self, X):
        self.ref_ = X.mean(axis=0)
        return self

    def transform(self, X):
        return self._msc_transform(X, self.ref_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @staticmethod
    def _msc_transform(X, ref):
        ref_mean = ref.mean()
        ref_centered = ref - ref_mean
        ref_var = np.mean(ref_centered ** 2)
        X_mean = X.mean(axis=1, keepdims=True)
        cov = np.mean((X - X_mean) * ref_centered, axis=1)
        b = cov / ref_var
        a = X_mean.flatten() - b * ref_mean
        b_safe = np.where(np.abs(b) > 1e-6, b, 1.0)
        return (X - a[:, None]) / b_safe[:, None]


class MSCV1:
    """MSC using the first right singular vector (v1) from uncentered SVD as reference."""
    def __init__(self):
        self.ref_ = None
        self.svd_random_state_ = 42

    def fit(self, X):
        X_f64 = X.astype(np.float64, copy=False)
        n_comp = min(N_DIM, min(X_f64.shape) - 1)
        _, _, Vt = randomized_svd(X_f64, n_components=n_comp,
                                  random_state=self.svd_random_state_)
        self.ref_ = Vt[0].copy()
        return self

    def transform(self, X):
        return MSCMean._msc_transform(X, self.ref_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MSCMedian:
    """MSC using the training-set median spectrum as reference."""
    def __init__(self):
        self.ref_ = None

    def fit(self, X):
        self.ref_ = np.median(X, axis=0)
        return self

    def transform(self, X):
        return MSCMean._msc_transform(X, self.ref_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_svm():
    return SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)


def method_a(X_train, X_test, y_train):
    """A (Proposed USP): Uncentered TruncSVD → SNV in PC space → StandardScaler → SVM."""
    svd = TruncatedSVD(n_components=N_DIM, random_state=42)
    Xtr_f = svd.fit_transform(X_train)
    Xte_f = svd.transform(X_test)

    Xtr_f = snv(Xtr_f)
    Xte_f = snv(Xte_f)

    sc = StandardScaler()
    clf = make_svm()
    clf.fit(sc.fit_transform(Xtr_f), y_train)
    return clf.predict(sc.transform(Xte_f))


def method_a2(X_train, X_test, y_train):
    """A2 (Conventional): SNV pre-processing → Centered TruncSVD → StandardScaler → SVM."""
    Xtr_snv = snv(X_train)
    Xte_snv = snv(X_test)

    mu = Xtr_snv.mean(axis=0)
    svd = TruncatedSVD(n_components=N_DIM, random_state=42)
    Xtr_f = svd.fit_transform(Xtr_snv - mu)
    Xte_f = svd.transform(Xte_snv - mu)

    sc = StandardScaler()
    clf = make_svm()
    clf.fit(sc.fit_transform(Xtr_f), y_train)
    return clf.predict(sc.transform(Xte_f))


def method_msc(X_train, X_test, y_train, msc_obj):
    """Generic MSC pipeline: MSC → Centered TruncSVD → StandardScaler → SVM."""
    Xtr_msc = msc_obj.fit_transform(X_train)
    Xte_msc = msc_obj.transform(X_test)

    mu = Xtr_msc.mean(axis=0)
    svd = TruncatedSVD(n_components=N_DIM, random_state=42)
    Xtr_f = svd.fit_transform(Xtr_msc - mu)
    Xte_f = svd.transform(Xte_msc - mu)

    sc = StandardScaler()
    clf = make_svm()
    clf.fit(sc.fit_transform(Xtr_f), y_train)
    return clf.predict(sc.transform(Xte_f)), msc_obj.ref_


def evaluate(y_true, y_pred):
    return {
        'OA': float(accuracy_score(y_true, y_pred)),
        'macro_F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
    }


def band_shift(X, shift=2):
    return np.roll(X, shift=shift, axis=1)


def run_msc_comparison(csv_path, out_dir=None):
    out_dir = Path(out_dir) if out_dir else BASE_DIR
    out_dir.mkdir(exist_ok=True, parents=True)

    X, y, wvl, meta = load_river_csv(csv_path, water_filter=0, band_range=(940, 1680))
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"Loaded: {n_samples} samples, {n_classes} classes, {n_features} bands")

    METHOD_NAMES = [
        "A (Proposed USP)",
        "A2 (SNV-Pre)",
        "A3_mean (MSC + mean ref)",
        "A3_v1 (MSC + v1 ref)",
        "A3_median (MSC + median ref)",
    ]

    records = []
    ref_spectra_all = {m: [] for m in METHOD_NAMES if 'MSC' in m}

    for rep in range(N_REPEATS):
        seed = SEED_START + rep
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # ── A (Proposed USP) ──
        y_pred_a = method_a(X_train, X_test, y_train)
        metrics_a = evaluate(y_test, y_pred_a)
        records.append({'repeat': rep, 'seed': seed, 'method': METHOD_NAMES[0], **metrics_a})

        # ── A2 (SNV-Pre) ──
        y_pred_a2 = method_a2(X_train, X_test, y_train)
        metrics_a2 = evaluate(y_test, y_pred_a2)
        records.append({'repeat': rep, 'seed': seed, 'method': METHOD_NAMES[1], **metrics_a2})

        # ── A3_mean (MSC + mean ref) ──
        y_pred_m1, ref_m1 = method_msc(X_train, X_test, y_train, MSCMean())
        metrics_m1 = evaluate(y_test, y_pred_m1)
        records.append({'repeat': rep, 'seed': seed, 'method': METHOD_NAMES[2], **metrics_m1})
        ref_spectra_all[METHOD_NAMES[2]].append(ref_m1)

        # ── A3_v1 (MSC + v1 ref) ──
        y_pred_v1, ref_v1 = method_msc(X_train, X_test, y_train, MSCV1())
        metrics_v1 = evaluate(y_test, y_pred_v1)
        records.append({'repeat': rep, 'seed': seed, 'method': METHOD_NAMES[3], **metrics_v1})
        ref_spectra_all[METHOD_NAMES[3]].append(ref_v1)

        # ── A3_median (MSC + median ref) ──
        y_pred_md, ref_md = method_msc(X_train, X_test, y_train, MSCMedian())
        metrics_md = evaluate(y_test, y_pred_md)
        records.append({'repeat': rep, 'seed': seed, 'method': METHOD_NAMES[4], **metrics_md})
        ref_spectra_all[METHOD_NAMES[4]].append(ref_md)

        if (rep + 1) % 5 == 0:
            print(f"  repeat {rep+1}/{N_REPEATS} done")

    df = pd.DataFrame(records)

    # ── Summary table ──
    method_order = {m: i for i, m in enumerate(METHOD_NAMES)}
    df['_order'] = df['method'].map(method_order)

    summary = df.groupby('method').agg(
        OA_mean=('OA', 'mean'),   OA_std=('OA', 'std'),
        F1_mean=('macro_F1', 'mean'), F1_std=('macro_F1', 'std'),
        kappa_mean=('kappa', 'mean'), kappa_std=('kappa', 'std'),
    ).reset_index()
    summary['_order'] = summary['method'].map(method_order)
    summary = summary.sort_values('_order').drop(columns=['_order'])

    print("\n" + "=" * 90)
    print("MSC Reference Comparison — Classification Results (mean ± std over 25 repeats)")
    print("=" * 90)
    header = f"{'Method':<32} {'OA':>15} {'Macro-F1':>15} {'Kappa':>15}"
    print(header)
    print("-" * 90)
    for _, row in summary.iterrows():
        oa_str = f"{row['OA_mean']*100:.1f} ± {row['OA_std']*100:.1f}"
        f1_str = f"{row['F1_mean']*100:.1f} ± {row['F1_std']*100:.1f}"
        kp_str = f"{row['kappa_mean']:.3f} ± {row['kappa_std']:.3f}"
        print(f"{row['method']:<32} {oa_str:>15} {f1_str:>15} {kp_str:>15}")

    # ── Cosine similarity between reference spectra ──
    print("\n" + "=" * 90)
    print("Cosine Similarity between Reference Spectra (averaged over 25 repeats)")
    print("=" * 90)

    msc_methods = [METHOD_NAMES[2], METHOD_NAMES[3], METHOD_NAMES[4]]
    pairs = [(0, 1, "mean vs v1"), (0, 2, "mean vs median"), (1, 2, "v1 vs median")]

    for i, j, label in pairs:
        cos_vals = []
        for k in range(N_REPEATS):
            a = ref_spectra_all[msc_methods[i]][k]
            b = ref_spectra_all[msc_methods[j]][k]
            cos_vals.append(cosine_similarity(a, b))
        cos_mean = np.mean(cos_vals)
        cos_std = np.std(cos_vals)
        print(f"  {label:<20}: {cos_mean:.4f} ± {cos_std:.4f}")

    # ── BandShift test (only if A3_v1 improves over A3_mean) ──
    v1_oa = summary[summary['method'] == METHOD_NAMES[3]]['OA_mean'].values[0]
    mean_oa = summary[summary['method'] == METHOD_NAMES[2]]['OA_mean'].values[0]

    if v1_oa > mean_oa:
        print("\n" + "=" * 90)
        print("BandShift Robustness Test (A3_v1 > A3_mean, running extra evaluation)")
        print("=" * 90)

        bs_records = []
        for rep in range(N_REPEATS):
            seed = SEED_START + rep
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed
            )
            X_train = X_train.astype(np.float32)
            X_test_bs = band_shift(X_test.astype(np.float32))

            for msc_cls, mname in [(MSCMean, METHOD_NAMES[2]),
                                    (MSCV1, METHOD_NAMES[3]),
                                    (MSCMedian, METHOD_NAMES[4])]:
                y_pred, _ = method_msc(X_train, X_test_bs, y_train, msc_cls())
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                bs_records.append({'repeat': rep, 'seed': seed, 'method': mname, 'macro_F1': f1})

        df_bs = pd.DataFrame(bs_records)
        bs_summary = df_bs.groupby('method')['macro_F1'].agg(['mean', 'std']).reset_index()
        bs_summary['_order'] = bs_summary['method'].map(method_order)
        bs_summary = bs_summary.sort_values('_order').drop(columns=['_order'])

        header_bs = f"{'Method':<32} {'Macro-F1 (BandShift)':>25}"
        print(header_bs)
        print("-" * 60)
        for _, row in bs_summary.iterrows():
            print(f"{row['method']:<32} {row['mean']*100:>10.1f} ± {row['std']*100:.1f}")
    else:
        print("\n[Info] A3_v1 does not improve over A3_mean; skipping BandShift test.")

    # ── Save results ──
    summary.to_csv(out_dir / "results_msc_comparison.csv", index=False, float_format="%.4f")
    print(f"\nResults saved to {out_dir / 'results_msc_comparison.csv'}")

    return df, summary


if __name__ == "__main__":
    csv_path = BASE_DIR / "1" / "spectradictionary.csv"
    if not csv_path.exists():
        print(f"ERROR: Data file not found: {csv_path}")
        print("Please download spectradictionary.csv from:")
        print("  https://zenodo.org/records/13377060")
        print("and place it in the 1/ directory.")
        sys.exit(1)

    run_msc_comparison(csv_path)
