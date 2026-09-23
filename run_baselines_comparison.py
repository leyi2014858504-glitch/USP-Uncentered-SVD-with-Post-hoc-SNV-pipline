#!/usr/bin/env python3
"""
run_baselines_comparison.py
===========================
Reviewer-requested fair baseline comparison + multi-condition + robustness.

Five methods compared at the same level (no chaining):
  1. USP       (proposed)  : Raw → Uncentered SVD → post-SNV in PC space
  2. SNV       (traditional): SNV → Centered SVD (PCA)
  3. SG1       (traditional): SG 1st derivative → Centered SVD (PCA)
  4. EMSC      (traditional): EMSC → Centered SVD (PCA)
  5. Detrend   (traditional): Polynomial detrend → Centered SVD (PCA)

Experiments:
  Phase 1 — Multi-condition comparison
    Three water conditions (Clear / Turbid / Foamy), each with 25 stratified
    4:1 repeats (seed 42..66). Reports OA / Macro-F1 / Kappa.

  Phase 2 — BandShift robustness (Clear water only)
    Same trained models from Phase 1 Clear, evaluated on np.roll(X, shift=2,
    axis=1) test set. Reports ΔOA drop per method.

Classifiers:
  a. LinearSVM   (kernel='linear', C=1.0)
  b. RBF-SVM     (kernel='rbf', C=1.0, gamma='scale')
  c. RandomForest(n_estimators=500)

Statistical tests:
  - RM-ANOVA + Bonferroni post-hoc on Clear water (5 methods, 25 repeats)
  - Paired t-test for BandShift vs Clear robustness

Outputs:
  results_baselines_raw.csv      — per-repeat raw data
  results_baselines_full.csv     — aggregated summary
  results_rf_importances.csv     — RF feature importances (PC1-PC20)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"
N_DIM = 20
N_REPEATS = 25
SEED_START = 42


# ───────────────────────── Data ─────────────────────────

WATER_MAP = {"clear": 0, "turbid": 1, "foamy": 2}
WATER_LABELS = {0: "Clear", 1: "Turbid", 2: "Foamy"}


def load_data(csv_path, water="clear"):
    """Load plastic dataset. water: 'clear' | 'turbid' | 'foamy' | 'all'."""
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["code", "fraction"] + list(range(350, 2501))
    df["material"] = (df["code"] // 10).astype(int)
    df["water"] = (df["code"] % 10).astype(int)
    df = df[df["material"] != 6]            # exclude Mix

    if water != "all":
        df = df[df["water"] == WATER_MAP[water]]

    band_cols = list(range(940, 1681))
    X = df[band_cols].values.astype(np.float32)
    y = df["material"].values.astype(np.int32)
    return X, y


# ───────────────────────── Preprocessing ─────────────────────────

def snv(X, eps=1e-8):
    """Row-wise SNV: per-sample mean/std normalization."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def sg_first_derivative(X, window=15, polyorder=2):
    """Savitzky-Golay 1st derivative (also smooths internally)."""
    return savgol_filter(X, window_length=window, polyorder=polyorder,
                         deriv=1, delta=1.0, axis=1).astype(np.float32)


def emsc(X_train, X_test=None, poly_order=2):
    """
    Extended MSC: x ≈ b0 + b_ref * r + b1*w + b2*w^2 + ...
    Correction: x_corrected = (x - b0 - b1*w - ... - bd*w^d) / b_ref
    Reference r = mean of training set.
    """
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
    """
    Standalone polynomial detrending (no SNV).
    Fit polynomial of degree poly_order to raw spectrum, subtract trend.
    """
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
    elif clf_type == 'rf':
        return RandomForestClassifier(n_estimators=500, max_depth=None,
                                      min_samples_leaf=1,
                                      class_weight='balanced_subsample',
                                      random_state=42, n_jobs=-1)
    else:  # 'svm' (linear)
        return SVC(kernel='linear', C=1.0, class_weight='balanced',
                  random_state=42)


CLASSIFIERS = ["svm", "rbf_svm", "rf"]


# ───────────────────────── Methods (fair comparison) ─────────────────────────
# Each method is a (method_name, preprocess_name, variant) tuple.
#   variant = "A_postSNV"    : Uncentered SVD → SNV in PC space (proposed USP)
#             "A2_noPostSNV" : Centered SVD (traditional PCA, no post-SNV)
# USP uses raw spectra (no preprocessing); the other 4 are traditional
# preprocessing → PCA pipelines.  No chaining (preprocessing + USP).

METHODS = [
    ("USP",      "none",     "A_postSNV"),     # Proposed
    ("SNV",      "SNV",      "A2_noPostSNV"),  # Traditional baseline
    ("SG1",      "SG1",      "A2_noPostSNV"),
    ("EMSC",     "EMSC",     "A2_noPostSNV"),
    ("Detrend",  "Detrend",  "A2_noPostSNV"),
]


# ───────────────────────── Pipelines ─────────────────────────

def pipeline_post_snv(Xtr_pre, Xte_pre, ytr, clf_type):
    """A-style (USP): TruncSVD(k=20) → SNV in PC space → StandardScaler → clf."""
    svd = TruncatedSVD(n_components=N_DIM, random_state=42)
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
    """A2-style (traditional): preprocessed → Centered TruncSVD → StandardScaler → clf."""
    mu = Xtr_pre.mean(axis=0)
    svd = TruncatedSVD(n_components=N_DIM, random_state=42)
    Ztr = svd.fit_transform(Xtr_pre - mu)
    Zte = svd.transform(Xte_pre - mu)
    sc = StandardScaler().fit(Ztr)
    clf = make_clf(clf_type)
    clf.fit(sc.transform(Ztr), ytr)
    y_pred = clf.predict(sc.transform(Zte))
    return y_pred, svd, clf, sc, mu


def apply_transform_bs(Xte_pre_bs, svd, sc, variant, mu=None):
    """Apply the trained pipeline to a (possibly perturbed) test set.
    Used for BandShift evaluation."""
    if variant == "A_postSNV":
        Zte = svd.transform(Xte_pre_bs)
        Zte = snv(Zte)
        return sc.transform(Zte)
    else:  # A2_noPostSNV
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
    """Shift spectra by `shift` bands along wavelength axis."""
    return np.roll(X, shift=shift, axis=1).astype(np.float32)


# ───────────────────────── Statistical Tests ─────────────────────────

def _f_sf(F, dfn, dfd):
    """Survival function for F-distribution."""
    try:
        from scipy.special import betainc
        x = dfd / (dfd + dfn * F)
        return float(betainc(dfd / 2.0, dfn / 2.0, x))
    except Exception:
        if F < 0.01:
            return 1.0
        z = (F ** (1.0 / 3.0) * (1.0 - 2.0 / (9.0 * dfd)) - (1.0 - 2.0 / (9.0 * dfn))) \
            / np.sqrt(2.0 / (9.0 * dfn) + F ** (2.0 / 3.0) * 2.0 / (9.0 * dfd))
        from math import erfc
        return 0.5 * erfc(z / np.sqrt(2.0))


def _t_sf(t, df):
    """Two-tailed survival function for t-distribution."""
    try:
        from scipy.stats import t as t_dist
        return 2.0 * float(t_dist.sf(abs(t), df))
    except Exception:
        from math import erfc
        return float(erfc(abs(t) / np.sqrt(2.0)))


def rm_anova(data_matrix):
    """Repeated-measures one-way ANOVA."""
    n, k = data_matrix.shape
    grand_mean = data_matrix.mean()
    subject_means = data_matrix.mean(axis=1)
    treatment_means = data_matrix.mean(axis=0)

    ss_total = ((data_matrix - grand_mean) ** 2).sum()
    ss_subjects = k * ((subject_means - grand_mean) ** 2).sum()
    ss_treatment = n * ((treatment_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_subjects - ss_treatment

    df_tr = k - 1
    df_err = (n - 1) * (k - 1)

    if ss_error < 1e-12:
        return float('inf'), 0.0, ss_treatment, ss_error, df_tr, df_err

    ms_treatment = ss_treatment / df_tr
    ms_error = ss_error / df_err
    F = ms_treatment / ms_error
    p_value = _f_sf(F, df_tr, df_err)
    return F, p_value, ss_treatment, ss_error, df_tr, df_err


def posthoc_bonferroni(data_matrix, method_names, alpha=0.05):
    """Paired t-tests with Bonferroni correction."""
    n, k = data_matrix.shape
    n_comparisons = k * (k - 1) // 2
    alpha_corrected = alpha / n_comparisons

    results = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = data_matrix[:, i] - data_matrix[:, j]
            mean_diff = diff.mean()
            se_diff = diff.std(ddof=1) / np.sqrt(n)
            if se_diff < 1e-12:
                t_stat = 0.0
                p_unc = 1.0
            else:
                t_stat = mean_diff / se_diff
                p_unc = _t_sf(t_stat, n - 1)
            p_corr = min(p_unc * n_comparisons, 1.0)
            results.append({
                'pair': f"{method_names[i]} vs {method_names[j]}",
                'method_i': method_names[i],
                'method_j': method_names[j],
                'mean_diff': mean_diff,
                't': t_stat,
                'p_uncorrected': p_unc,
                'p_corrected': p_corr,
                'significant': p_corr < alpha,
            })
    return results


def run_stats(df_raw, clf_type, alpha=0.05):
    """Run RM-ANOVA + Bonferroni post-hoc for one classifier."""
    sub = df_raw[df_raw['classifier'] == clf_type]
    if sub.empty:
        return None

    method_order = [m[0] for m in METHODS]  # ["USP", "SNV", "SG1", "EMSC", "Detrend"]
    pivot = sub.pivot_table(index='repeat', columns='method',
                            values='OA', aggfunc='first')
    pivot = pivot[method_order]
    data_mat = pivot.values  # (n_repeats, 5)

    F, p, ss_tr, ss_err, df_tr, df_err = rm_anova(data_mat)

    clf_label = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM",
                 "rf": "RandomForest"}[clf_type]

    print(f"\n{'='*80}")
    print(f"[{clf_label}]  Repeated-Measures ANOVA")
    print(f"{'='*80}")
    print(f"  F({df_tr}, {df_err}) = {F:.4f},  p = {p:.6f}  "
          f"{'*** SIGNIFICANT' if p < alpha else '(n.s.)'}")

    posthoc = posthoc_bonferroni(data_mat, method_order, alpha=alpha)
    sig_count = sum(1 for ph in posthoc if ph['significant'])

    print(f"\n  Post-hoc (Bonferroni, α_corrected = {alpha/10:.4f}):")
    print(f"  {'Pair':<30} {'Mean Δ':>8} {'t':>8} {'p(unc)':>10} {'p(corr)':>10} {'Sig':>6}")
    print(f"  {'-'*70}")
    for ph in posthoc:
        sig_mark = "  ***" if ph['significant'] else ""
        print(f"  {ph['pair']:<30} {ph['mean_diff']*100:>7.2f}% "
              f"{ph['t']:>8.3f} {ph['p_uncorrected']:>10.6f} "
              f"{ph['p_corrected']:>10.6f}{sig_mark}")

    return {'F': F, 'p_anova': p, 'posthoc': posthoc, 'n_significant': sig_count}


# ───────────────────────── Main runner ─────────────────────────

def _run_one_condition(X, y, condition_label, records, rf_imp_rows,
                       robustness=False):
    """Run all 5 methods × 3 classifiers for one water condition.
    If robustness=True, also evaluate on BandShift test set (same trained models).
    """
    for rep in range(N_REPEATS):
        seed = SEED_START + rep
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        Xtr = Xtr.astype(np.float32)
        Xte = Xte.astype(np.float32)

        Xte_bs = bandshift(Xte, shift=2) if robustness else None

        for method_name, pp_name, variant in METHODS:
            pp_fn = PREPROCESSORS[pp_name]
            Xtr_pp, Xte_pp = pp_fn(Xtr, Xte)
            Xte_pp_bs = pp_fn(Xtr, Xte_bs)[1] if Xte_bs is not None else None

            for clf_type in CLASSIFIERS:
                # Normal test
                if variant == "A_postSNV":
                    y_pred, svd, clf, sc = pipeline_post_snv(
                        Xtr_pp, Xte_pp, ytr, clf_type
                    )
                    mu = None
                else:
                    y_pred, svd, clf, sc, mu = pipeline_pre_snv(
                        Xtr_pp, Xte_pp, ytr, clf_type
                    )
                metrics = evaluate(yte, y_pred)
                records.append({
                    'condition': condition_label,
                    'method': method_name,
                    'repeat': rep, 'seed': seed,
                    'preprocess': pp_name,
                    'variant': variant,
                    'classifier': clf_type,
                    **metrics,
                })

                # BandShift test (use the SAME trained model)
                if Xte_bs is not None:
                    Xte_final_bs = apply_transform_bs(Xte_pp_bs, svd, sc,
                                                       variant, mu=mu)
                    y_pred_bs = clf.predict(Xte_final_bs)
                    metrics_bs = evaluate(yte, y_pred_bs)
                    records.append({
                        'condition': f"{condition_label}+BandShift",
                        'method': method_name,
                        'repeat': rep, 'seed': seed,
                        'preprocess': pp_name,
                        'variant': variant,
                        'classifier': clf_type,
                        **metrics_bs,
                    })

                # Save RF feature importances for the first repeat (normal test only)
                if rep == 0 and clf_type == 'rf' and not robustness:
                    if hasattr(clf, 'feature_importances_'):
                        for k in range(N_DIM):
                            rf_imp_rows.append({
                                'condition': condition_label,
                                'method': method_name,
                                'component': f"PC{k+1}",
                                'importance': float(clf.feature_importances_[k]),
                            })

        if (rep + 1) % 5 == 0 or rep == 0:
            print(f"  [{condition_label}] repeat {rep+1}/{N_REPEATS} done")


def _print_summary_table(summary, condition_label):
    """Print 5-method summary for one condition."""
    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM", "rf": "RandomForest"}
    sub = summary[summary['condition'] == condition_label]
    if sub.empty:
        return

    print(f"\n--- {condition_label} ---")
    header = (f"{'Method':<12} {'Classifier':<14} {'OA':>18} "
              f"{'Macro-F1':>18} {'Kappa':>14}")
    print(header)
    print("-" * 80)
    for method_name, _, _ in METHODS:
        for clf in CLASSIFIERS:
            row = sub[(sub['method'] == method_name) &
                      (sub['classifier'] == clf)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            oa = f"{r['OA_mean']*100:.2f} ± {r['OA_std']*100:.2f}"
            f1 = f"{r['F1_mean']*100:.2f} ± {r['F1_std']*100:.2f}"
            kp = f"{r['k_mean']:.4f} ± {r['k_std']:.4f}"
            print(f"{method_name:<12} {clf_labels[clf]:<14} {oa:>18} "
                  f"{f1:>18} {kp:>14}")
        print()


def run():
    csv_path = DATA_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Download from https://zenodo.org/records/13377060")
        return

    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM", "rf": "RandomForest"}

    # ─── Phase 1: Multi-condition ───
    print("=" * 100)
    print("Phase 1: Multi-condition fair comparison (Clear / Turbid / Foamy)")
    print("        Clear water also runs BandShift robustness test.")
    print("        5 methods × 3 classifiers = 15 experiments per condition")
    print("=" * 100)

    records = []
    rf_imp_rows = []

    for water_key, water_label in [("clear", "Clear"),
                                    ("turbid", "Turbid"),
                                    ("foamy", "Foamy")]:
        X, y = load_data(csv_path, water=water_key)
        print(f"\n[{water_label}] {len(y)} samples | {X.shape[1]} bands | "
              f"{len(np.unique(y))} classes")
        do_robust = (water_key == "clear")
        _run_one_condition(X, y, water_label, records, rf_imp_rows,
                           robustness=do_robust)

    # ─── Save raw + summary ───
    df = pd.DataFrame(records)
    out_raw = BASE_DIR / "results_baselines_raw.csv"
    df.to_csv(out_raw, index=False, float_format="%.6f")
    print(f"\nSaved raw per-repeat data: {out_raw}")

    summary = df.groupby(
        ['condition', 'method', 'classifier']
    ).agg(
        OA_mean=('OA', 'mean'), OA_std=('OA', 'std'),
        F1_mean=('macro_F1', 'mean'), F1_std=('macro_F1', 'std'),
        k_mean=('kappa', 'mean'), k_std=('kappa', 'std'),
    ).reset_index()
    out_full = BASE_DIR / "results_baselines_full.csv"
    summary.to_csv(out_full, index=False, float_format="%.4f")
    print(f"Saved full summary: {out_full}")

    if rf_imp_rows:
        df_imp = pd.DataFrame(rf_imp_rows)
        out_imp = BASE_DIR / "results_rf_importances.csv"
        df_imp.to_csv(out_imp, index=False, float_format="%.6f")
        print(f"Saved RF importances: {out_imp}")

    # ─── Print summary tables ───
    for cond in ["Clear", "Turbid", "Foamy"]:
        print("\n" + "=" * 100)
        print(f"Fair Comparison — {cond} water")
        print("=" * 100)
        _print_summary_table(summary, cond)

    # ─── Robustness table ───
    print("\n" + "=" * 100)
    print("Robustness: BandShift (shift=2) on Clear water")
    print("=" * 100)
    _print_summary_table(summary, "Clear+BandShift")

    # ─── Drop in accuracy ───
    print("\n" + "=" * 100)
    print("Robustness drop: ΔOA = OA(BandShift) − OA(Clear) for each method")
    print("=" * 100)
    print(f"{'Method':<12} {'Classifier':<14} {'OA(Clear)':>10} "
          f"{'OA(BS)':>10} {'Δ OA (pp)':>12}")
    print("-" * 60)
    for method_name, _, _ in METHODS:
        for clf in CLASSIFIERS:
            row_c = summary[(summary['condition'] == 'Clear') &
                            (summary['method'] == method_name) &
                            (summary['classifier'] == clf)]
            row_b = summary[(summary['condition'] == 'Clear+BandShift') &
                            (summary['method'] == method_name) &
                            (summary['classifier'] == clf)]
            if len(row_c) == 0 or len(row_b) == 0:
                continue
            rc, rb = row_c.iloc[0], row_b.iloc[0]
            delta = (rb['OA_mean'] - rc['OA_mean']) * 100
            print(f"{method_name:<12} {clf_labels[clf]:<14} "
                  f"{rc['OA_mean']*100:>9.2f}% {rb['OA_mean']*100:>9.2f}% "
                  f"{delta:>11.2f}")

    # ─── Statistical tests (Clear water) ───
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Clear water)")
    print("=" * 80)
    print("Repeated-measures ANOVA + Bonferroni post-hoc")
    print("(5 methods as within-subject factor, 25 repeats)")

    df_clear = df[df['condition'] == 'Clear']
    all_stats = {}
    for clf_type in CLASSIFIERS:
        all_stats[clf_type] = run_stats(df_clear, clf_type)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: RM-ANOVA F-values")
    print(f"{'='*80}")
    print(f"{'Classifier':<14} {'F(4,96)':>12} {'p':>12} {'Sig':>6}")
    print(f"{'-'*50}")
    for clf_type in CLASSIFIERS:
        if all_stats[clf_type] is not None:
            s = all_stats[clf_type]
            sig = "  ***" if s['p_anova'] < 0.05 else ""
            print(f"{clf_labels[clf_type]:<14} {s['F']:>12.4f} "
                  f"{s['p_anova']:>12.6f}{sig}")

    # ─── Robustness stats ───
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS: BandShift vs Clear (paired)")
    print("=" * 80)
    df_bs = df[df['condition'] == 'Clear+BandShift']
    n_comparisons = 5 * 3  # 5 methods × 3 classifiers = 15
    alpha_corr = 0.05 / n_comparisons
    print(f"{'Method':<12} {'Classifier':<14} {'t':>8} {'p(unc)':>10} "
          f"{'p(corr)':>10} {'Sig':>6}")
    print("-" * 60)
    for method_name, _, _ in METHODS:
        for clf in CLASSIFIERS:
            sub_c = df_clear[(df_clear['method'] == method_name) &
                             (df_clear['classifier'] == clf)].sort_values('repeat')
            sub_b = df_bs[(df_bs['method'] == method_name) &
                          (df_bs['classifier'] == clf)].sort_values('repeat')
            if len(sub_c) == 0 or len(sub_b) == 0:
                continue
            d = sub_c.OA.values - sub_b.OA.values
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 1e-12 else 0.0
            p_u = _t_sf(t, len(d) - 1) if se > 1e-12 else 1.0
            p_c = min(p_u * n_comparisons, 1.0)
            sig = "  ***" if p_c < 0.05 else ""
            print(f"{method_name:<12} {clf_labels[clf]:<14} {t:>8.3f} "
                  f"{p_u:>10.6f} {p_c:>10.6f}{sig}")
    print(f"\nα_corrected = 0.05 / {n_comparisons} = {alpha_corr:.4f}")


if __name__ == "__main__":
    run()
