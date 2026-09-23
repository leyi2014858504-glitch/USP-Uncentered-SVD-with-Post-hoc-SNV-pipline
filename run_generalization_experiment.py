#!/usr/bin/env python3
"""
run_generalization_experiment.py
================================
Generalization experiment on a new NIR/SWIR plastic dataset.

This dataset is structurally different from the original plastic dataset:
  - 228 samples (vs 193 in Clear water original)
  - Wavelength range 850-2400 nm (vs 940-1680 nm)
  - 6 polymers: PE, PET, PLA, PP, PVC, SBR
  - 6 water conditions: Bu5, W6, TG, C, V, DIW
  - 6 concentrations: 0.15%, 0.5%, 1.5%, 5%, 15%, 50%
  - Plus 6 pure-water backgrounds (skipped) and 6 pure-plastic samples (kept)

Tests whether the USP framework (proposed) generalizes beyond the original
plastic dataset to a different sensor, wavelength range, and sample
distribution.

Methods compared (5, fair — same as run_baselines_comparison.py):
  1. USP       (proposed)  : Raw → Uncentered SVD → post-SNV in PC space
  2. SNV       (traditional): SNV → Centered SVD (PCA)
  3. SG1       (traditional): SG 1st derivative → Centered SVD (PCA)
  4. EMSC      (traditional): EMSC → Centered SVD (PCA)
  5. Detrend   (traditional): Polynomial detrend → Centered SVD (PCA)

Classifiers:
  a. LinearSVM   b. RBF-SVM   (RF dropped — intrinsic BandShift robustness confounds)

Setup:
  - 5-fold stratified CV × 5 seeds (seed 42..46) = 25 evaluations
  - Wavelength range: 940-1680 nm (overlap with original plastic dataset)
  - 6-class classification

Experiments:
  Phase 1: All usable samples (pooled) — main result
  Phase 2: Per-condition (6 water types) — cross-condition robustness
  Phase 3: Cross-condition transfer (train on 5, test on 1 held-out)
  Phase 4: BandShift robustness on pooled data

Statistical tests:
  - RM-ANOVA + Bonferroni post-hoc on pooled data
  - Paired t-test for cross-condition transfer

Outputs:
  results_generalization_raw.csv      — per-repeat raw data
  results_generalization_full.csv     — aggregated summary
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.signal import savgol_filter
from scipy.special import betainc
from scipy.stats import t as tdist

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"
N_DIM = 20
SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 5  # 5-fold CV × 5 seeds = 25 evaluations


# ───────────────────────── Data ─────────────────────────

POLYMER_LABELS = {0: "PE", 1: "PET", 2: "PLA", 3: "PP", 4: "PVC", 5: "SBR"}
CONDITION_LABELS = {
    "Bu5": "Bu5", "W6": "W6", "TG": "TG",
    "C": "C (clear)", "V": "V", "DIW": "DIW",
}


def _parse_conc(s):
    """Convert concentration strings: '0X15' → 0.15, '1X5' → 1.5, '50' → 50."""
    if 'X' in s:
        return float(s.replace('X', '.'))
    return float(s)


def load_generalization_data(csv_path, band_range=(940, 1680),
                              condition_filter=None,
                              include_pure_plastic=True,
                              include_background=False):
    """
    Load the generalization dataset.

    Format (transposed from spectradictionary.csv):
      rows = wavelengths, columns = samples
      Column naming: <condition>_<polymer>_<concentration>
      Special: <cond>__0 = pure water (no plastic)
              _<polymer>_100 = pure plastic (no condition)

    Parameters
    ----------
    band_range : (lo, hi)
        Wavelength range to keep (default 940-1680 nm, matching original).
    condition_filter : str or None
        If set, only samples from this condition are returned (pure plastics
        always included if `include_pure_plastic=True`).
    include_pure_plastic : bool
        Whether to include pure-plastic samples (100% plastic).
    include_background : bool
        Whether to include pure-water background samples (label = "background").
        Default False — these have no plastic type to classify.

    Returns
    -------
    X : (n_samples, n_bands) float32
    y : (n_samples,) int32  — polymer label
    wvl : (n_bands,) float32
    meta : DataFrame with columns [condition, polymer, conc, sample_id]
    """
    df = pd.read_csv(csv_path)

    # Filter by wavelength
    wvl = df['wavelength'].values.astype(np.float32)
    mask = (wvl >= band_range[0]) & (wvl <= band_range[1])
    wvl = wvl[mask]
    X_T = df.iloc[mask, 1:].values  # (n_bands, n_samples)

    cols = list(df.columns[1:])

    samples, labels = [], []
    conds, polys, concs, ids = [], [], [], []

    for i, col in enumerate(cols):
        parts = col.split('_')

        # Pure water background: <cond>__0
        if len(parts) == 3 and parts[1] == '' and parts[2] == '0':
            if not include_background:
                continue
            cond, poly, conc = parts[0], 'background', 0.0
        # Pure plastic: _<poly>_100
        elif len(parts) == 3 and parts[0] == '' and parts[2] == '100':
            if not include_pure_plastic:
                continue
            cond, poly, conc = 'Pure', parts[1], 100.0
        # Regular sample: <cond>_<poly>_<conc>
        elif len(parts) == 3 and parts[1] in ('PE', 'PET', 'PLA', 'PP', 'PVC', 'SBR'):
            cond, poly, conc = parts[0], parts[1], _parse_conc(parts[2])
        else:
            continue

        if condition_filter and cond != condition_filter and cond != 'Pure':
            continue

        samples.append(X_T[:, i])
        labels.append(poly)
        conds.append(cond)
        polys.append(poly)
        concs.append(conc)
        ids.append(col)

    X = np.array(samples, dtype=np.float32)
    label_map = {p: i for i, p in enumerate(sorted(set(labels)))}
    y = np.array([label_map[l] for l in labels], dtype=np.int32)

    meta = pd.DataFrame({
        'sample_id': ids,
        'condition': conds,
        'polymer': polys,
        'conc': concs,
    })

    return X, y, wvl, meta, label_map


# ───────────────────────── Preprocessing ─────────────────────────
# Same implementations as run_baselines_comparison.py (no chaining).

def snv(X, eps=1e-8):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def sg_first_derivative(X, window=15, polyorder=2):
    return savgol_filter(X, window_length=window, polyorder=polyorder,
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
    elif clf_type == 'rf':
        return RandomForestClassifier(n_estimators=500, max_depth=None,
                                      min_samples_leaf=1,
                                      class_weight='balanced_subsample',
                                      random_state=42, n_jobs=-1)
    else:
        return SVC(kernel='linear', C=1.0, class_weight='balanced',
                  random_state=42)


CLASSIFIERS = ["svm", "rbf_svm"]  # RF dropped (intrinsic BandShift robustness confounds)


# ───────────────────────── Methods ─────────────────────────
# Same 5-method fair comparison as run_baselines_comparison.py.

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

def rm_anova(mat):
    n, k = mat.shape
    gm, sm, tm = mat.mean(), mat.mean(1), mat.mean(0)
    ss_tr = n * ((tm - gm) ** 2).sum()
    ss_sub = k * ((sm - gm) ** 2).sum()
    ss_tot = ((mat - gm) ** 2).sum()
    ss_err = ss_tot - ss_sub - ss_tr
    df_tr, df_err = k - 1, (n - 1) * (k - 1)
    if ss_err < 1e-12:
        return float('inf'), 0.0, df_tr, df_err
    F = (ss_tr / df_tr) / (ss_err / df_err)
    x = df_err / (df_err + df_tr * F)
    p = float(betainc(df_err / 2.0, df_tr / 2.0, x))
    return F, p, df_tr, df_err


def posthoc_bonferroni(mat, names, alpha=0.05):
    n, k = mat.shape
    nc = k * (k - 1) // 2
    ac = alpha / nc
    res = []
    for i in range(k):
        for j in range(i + 1, k):
            d = mat[:, i] - mat[:, j]
            se = d.std(ddof=1) / np.sqrt(n)
            t = d.mean() / se if se > 1e-12 else 0.0
            pu = 2.0 * tdist.sf(abs(t), n - 1) if se > 1e-12 else 1.0
            pc = min(pu * nc, 1.0)
            res.append((f"{names[i]} vs {names[j]}", d.mean(), t, pu, pc, pc < ac))
    return res, ac


# ───────────────────────── Phase runners ─────────────────────────

def run_phase1_pooled(X, y, records, rf_imp_rows):
    """Phase 1: Pooled 5-fold CV × 5 seeds + BandShift."""
    print("\n[Phase 1] Pooled data, 5-fold CV × 5 seeds × 5 methods × 2 classifiers")
    eval_idx = 0
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            Xtr = X[train_idx].astype(np.float32)
            Xte = X[test_idx].astype(np.float32)
            ytr = y[train_idx]
            yte = y[test_idx]

            Xte_bs = bandshift(Xte, shift=2)

            for method_name, pp_name, variant in METHODS:
                pp_fn = PREPROCESSORS[pp_name]
                Xtr_pp, Xte_pp = pp_fn(Xtr, Xte)
                Xte_pp_bs = pp_fn(Xtr, Xte_bs)[1]

                for clf_type in CLASSIFIERS:
                    if variant == "A_postSNV":
                        y_pred, svd, clf, sc = pipeline_post_snv(
                            Xtr_pp, Xte_pp, ytr, clf_type)
                        mu = None
                    else:
                        y_pred, svd, clf, sc, mu = pipeline_pre_snv(
                            Xtr_pp, Xte_pp, ytr, clf_type)
                    metrics = evaluate(yte, y_pred)
                    records.append({
                        'phase': 1, 'condition': 'Pooled',
                        'method': method_name,
                        'repeat': eval_idx, 'seed': seed, 'fold': fold_idx,
                        'preprocess': pp_name, 'variant': variant,
                        'classifier': clf_type, **metrics,
                    })

                    # BandShift
                    Xte_final_bs = apply_transform_bs(Xte_pp_bs, svd, sc, variant, mu=mu)
                    y_pred_bs = clf.predict(Xte_final_bs)
                    metrics_bs = evaluate(yte, y_pred_bs)
                    records.append({
                        'phase': 1, 'condition': 'Pooled+BandShift',
                        'method': method_name,
                        'repeat': eval_idx, 'seed': seed, 'fold': fold_idx,
                        'preprocess': pp_name, 'variant': variant,
                        'classifier': clf_type, **metrics_bs,
                    })

            eval_idx += 1

        print(f"  seed {seed} done ({eval_idx}/{N_SPLITS * len(SEEDS)} evals)")


def run_phase2_per_condition(X, y, meta, records):
    """Phase 2: Per-condition experiments (6 water types, 5-fold CV × 5 seeds)."""
    print("\n[Phase 2] Per-condition experiments (5-fold CV × 5 seeds)")
    for cond in ['Bu5', 'W6', 'TG', 'C', 'V', 'DIW']:
        mask = (meta['condition'] == cond) | (meta['condition'] == 'Pure')
        if mask.sum() < 30:
            print(f"  [{cond}] only {mask.sum()} samples, skipping")
            continue

        Xc = X[mask]
        yc = y[mask]
        n_classes = len(np.unique(yc))
        if n_classes < 2 or mask.sum() < n_classes * N_SPLITS:
            print(f"  [{cond}] insufficient classes/samples for {N_SPLITS}-fold, skipping")
            continue

        print(f"  [{cond}] {mask.sum()} samples, {n_classes} classes")

        eval_idx = 0
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(Xc, yc)):
                Xtr = Xc[train_idx].astype(np.float32)
                Xte = Xc[test_idx].astype(np.float32)
                ytr = yc[train_idx]
                yte = yc[test_idx]

                for method_name, pp_name, variant in METHODS:
                    pp_fn = PREPROCESSORS[pp_name]
                    Xtr_pp, Xte_pp = pp_fn(Xtr, Xte)

                    for clf_type in CLASSIFIERS:
                        if variant == "A_postSNV":
                            y_pred, _, _, _ = pipeline_post_snv(
                                Xtr_pp, Xte_pp, ytr, clf_type)
                        else:
                            y_pred, _, _, _, _ = pipeline_pre_snv(
                                Xtr_pp, Xte_pp, ytr, clf_type)
                        metrics = evaluate(yte, y_pred)
                        records.append({
                            'phase': 2, 'condition': cond,
                            'method': method_name,
                            'repeat': eval_idx, 'seed': seed, 'fold': fold_idx,
                            'preprocess': pp_name, 'variant': variant,
                            'classifier': clf_type, **metrics,
                        })

                eval_idx += 1

            print(f"    seed {seed} done ({eval_idx}/{N_SPLITS * len(SEEDS)} evals)")


def run_phase3_cross_condition(X, y, meta, records):
    """Phase 3: Cross-condition transfer — train on 5 conditions, test on 1."""
    print("\n[Phase 3] Cross-condition transfer (train on 5, test on 1)")
    conditions = ['Bu5', 'W6', 'TG', 'C', 'V', 'DIW']

    for held_out in conditions:
        train_mask = (meta['condition'] != held_out) & \
                     (meta['condition'] != 'Pure')
        test_mask = (meta['condition'] == held_out)

        if train_mask.sum() < 30 or test_mask.sum() < 6:
            print(f"  [hold-out {held_out}] insufficient samples, skipping")
            continue

        Xtr = X[train_mask].astype(np.float32)
        ytr = y[train_mask]
        Xte = X[test_mask].astype(np.float32)
        yte = y[test_mask]

        print(f"  [hold-out {held_out}] train={train_mask.sum()}, "
              f"test={test_mask.sum()}")

        for method_name, pp_name, variant in METHODS:
            pp_fn = PREPROCESSORS[pp_name]
            Xtr_pp, Xte_pp = pp_fn(Xtr, Xte)

            for clf_type in CLASSIFIERS:
                if variant == "A_postSNV":
                    y_pred, _, _, _ = pipeline_post_snv(
                        Xtr_pp, Xte_pp, ytr, clf_type)
                else:
                    y_pred, _, _, _, _ = pipeline_pre_snv(
                        Xtr_pp, Xte_pp, ytr, clf_type)
                metrics = evaluate(yte, y_pred)
                records.append({
                    'phase': 3, 'condition': f"hold-out {held_out}",
                    'method': method_name,
                    'repeat': 0, 'seed': 42,
                    'preprocess': pp_name, 'variant': variant,
                    'classifier': clf_type, **metrics,
                })


# ───────────────────────── Reporting ─────────────────────────

def _print_table(summary, condition_label, phase=1):
    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}
    sub = summary[(summary['condition'] == condition_label) &
                  (summary['phase'] == phase)]
    if sub.empty:
        return

    print(f"\n--- {condition_label} ---")
    header = f"{'Method':<12} {'Classifier':<14} {'OA':>18} {'Macro-F1':>18} {'Kappa':>14}"
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
            print(f"{method_name:<12} {clf_labels[clf]:<14} {oa:>18} {f1:>18} {kp:>14}")
        print()


def _print_cross_condition_table(df_raw):
    """Phase 3 cross-condition results (single run, no std)."""
    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}
    sub = df_raw[df_raw['phase'] == 3]
    if sub.empty:
        return

    print("\n" + "=" * 100)
    print("Phase 3: Cross-condition transfer — OA on held-out condition")
    print("=" * 100)
    conditions = sorted(sub['condition'].unique())
    header = f"{'Method':<12} {'Classifier':<14}" + \
             ''.join(f'{c:>14}' for c in conditions) + f"{'Mean':>10}"
    print(header)
    print("-" * (12 + 14 + 14 * len(conditions) + 10))
    for method_name, _, _ in METHODS:
        for clf in CLASSIFIERS:
            row_vals = []
            for c in conditions:
                r = sub[(sub['method'] == method_name) &
                        (sub['classifier'] == clf) &
                        (sub['condition'] == c)]
                if len(r) == 0:
                    row_vals.append('—')
                else:
                    row_vals.append(f"{r.iloc[0]['OA']*100:.2f}%")
            mean_oa = sub[(sub['method'] == method_name) &
                          (sub['classifier'] == clf)]['OA'].mean()
            print(f"{method_name:<12} {clf_labels[clf]:<14}" +
                  ''.join(f'{v:>14}' for v in row_vals) +
                  f"{mean_oa*100:>9.2f}%")
        print()


def run_stats_phase1(df_pooled, alpha=0.05):
    """RM-ANOVA + Bonferroni on Phase 1 pooled results."""
    method_order = [m[0] for m in METHODS]
    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}

    print("\n" + "=" * 80)
    print("STATISTICAL TESTS — Phase 1 (Pooled)")
    print("=" * 80)

    for clf_type in CLASSIFIERS:
        sub = df_pooled[df_pooled['classifier'] == clf_type]
        piv = sub.pivot_table(index='repeat', columns='method',
                                values='OA', aggfunc='first')[method_order]
        mat = piv.values

        F, p, dft, dfe = rm_anova(mat)
        S = " *** SIG" if p < alpha else ""
        print(f"\n[{clf_labels[clf_type]}]  RM-ANOVA  F({dft},{dfe})={F:.4f}  "
              f"p={p:.6f}{S}")

        ph, ac = posthoc_bonferroni(mat, method_order, alpha=alpha)
        print(f"  Post-hoc (Bonferroni, α_corr={ac:.4f}):")
        print(f"  {'Pair':<26} {'Δ(OA)%':>8} {'t':>8} {'p_unc':>10} "
              f"{'p_corr':>10} {'Sig':>6}")
        print(f"  {'-'*68}")
        for pair, md, t, pu, pc, s in ph:
            sig = "  ***" if s else ""
            print(f"  {pair:<26} {md*100:>7.2f}% {t:>8.3f} "
                  f"{pu:>10.6f} {pc:>10.6f}{sig}")


def run_stats_bandshift(df_pooled, alpha=0.05):
    """Paired t-test: BandShift vs Clear (Pooled) per method/classifier."""
    method_order = [m[0] for m in METHODS]
    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}

    print("\n" + "=" * 80)
    print("STATISTICAL TESTS — BandShift vs Pooled (paired t-test)")
    print("=" * 80)
    df_clear = df_pooled  # phase 1, condition 'Pooled'
    df_bs = df_pooled.assign(
        condition=lambda d: d['condition'].replace('Pooled', 'Pooled+BandShift')
    )  # placeholder; use raw records instead

    # Pull from raw df
    pass


# ───────────────────────── Main ─────────────────────────

def run():
    csv_path = DATA_DIR / "__Mean_Spectra_Reflectance.csv"
    if not csv_path.exists():
        # Try root directory
        csv_path = BASE_DIR / "__Mean_Spectra_Reflectance.csv"
    if not csv_path.exists():
        print(f"ERROR: __Mean_Spectra_Reflectance.csv not found.")
        return

    print("=" * 100)
    print("Generalization Experiment on New NIR/SWIR Plastic Dataset")
    print("=" * 100)

    # ─── Load data ───
    X, y, wvl, meta, label_map = load_generalization_data(
        csv_path, band_range=(940, 1680),
        include_pure_plastic=True, include_background=False
    )

    print(f"\nDataset summary:")
    print(f"  Total samples: {len(y)}")
    print(f"  Wavelength range: {wvl[0]:.0f} - {wvl[-1]:.0f} nm ({len(wvl)} bands)")
    print(f"  Polymers: {label_map}")
    print(f"  Conditions: {sorted(meta['condition'].unique())}")
    print(f"  Polymer counts:")
    for cls in np.unique(y):
        print(f"    {POLYMER_LABELS.get(int(cls), cls)}: {(y == cls).sum()}")
    print(f"  Per-condition counts:")
    for cond in sorted(meta['condition'].unique()):
        n = (meta['condition'] == cond).sum()
        print(f"    {cond}: {n}")

    # ─── Run all phases ───
    records = []
    rf_imp_rows = []

    run_phase1_pooled(X, y, records, rf_imp_rows)
    run_phase2_per_condition(X, y, meta, records)
    run_phase3_cross_condition(X, y, meta, records)

    # ─── Save raw + summary ───
    df = pd.DataFrame(records)
    out_raw = BASE_DIR / "results_generalization_raw.csv"
    df.to_csv(out_raw, index=False, float_format="%.6f")
    print(f"\nSaved raw: {out_raw}")

    summary = df.groupby(['phase', 'condition', 'method', 'classifier']).agg(
        OA_mean=('OA', 'mean'), OA_std=('OA', 'std'),
        F1_mean=('macro_F1', 'mean'), F1_std=('macro_F1', 'std'),
        k_mean=('kappa', 'mean'), k_std=('kappa', 'std'),
    ).reset_index()
    out_full = BASE_DIR / "results_generalization_full.csv"
    summary.to_csv(out_full, index=False, float_format="%.4f")
    print(f"Saved summary: {out_full}")

    if rf_imp_rows:
        df_imp = pd.DataFrame(rf_imp_rows)
        out_imp = BASE_DIR / "results_generalization_rf_importances.csv"
        df_imp.to_csv(out_imp, index=False, float_format="%.6f")
        print(f"Saved RF importances: {out_imp}")

    # ─── Print summary tables ───
    print("\n" + "=" * 100)
    print("Phase 1: Pooled (all samples, 5-fold CV × 5 seeds)")
    print("=" * 100)
    _print_table(summary, 'Pooled', phase=1)

    print("\n" + "=" * 100)
    print("Phase 1: BandShift robustness (shift=2 bands)")
    print("=" * 100)
    _print_table(summary, 'Pooled+BandShift', phase=1)

    print("\n" + "=" * 100)
    print("Phase 2: Per-condition experiments")
    print("=" * 100)
    for cond in ['Bu5', 'W6', 'TG', 'C', 'V', 'DIW']:
        _print_table(summary, cond, phase=2)

    _print_cross_condition_table(df)

    # ─── BandShift drop summary ───
    print("\n" + "=" * 100)
    print("BandShift drop: ΔOA = OA(BandShift) − OA(Pooled) per method")
    print("=" * 100)
    clf_labels = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}
    print(f"{'Method':<12} {'Classifier':<14} {'OA(Pooled)':>11} {'OA(BS)':>9} {'Δ(pp)':>8}")
    print("-" * 60)
    for method_name, _, _ in METHODS:
        for clf in CLASSIFIERS:
            row_c = summary[(summary['condition'] == 'Pooled') &
                            (summary['method'] == method_name) &
                            (summary['classifier'] == clf)]
            row_b = summary[(summary['condition'] == 'Pooled+BandShift') &
                            (summary['method'] == method_name) &
                            (summary['classifier'] == clf)]
            if len(row_c) == 0 or len(row_b) == 0:
                continue
            rc, rb = row_c.iloc[0], row_b.iloc[0]
            delta = (rb['OA_mean'] - rc['OA_mean']) * 100
            print(f"{method_name:<12} {clf_labels[clf]:<14} "
                  f"{rc['OA_mean']*100:>10.2f}% {rb['OA_mean']*100:>8.2f}% "
                  f"{delta:>7.2f}")

    # ─── Statistical tests on Phase 1 ───
    df_pooled = df[(df['phase'] == 1) & (df['condition'] == 'Pooled')]
    run_stats_phase1(df_pooled)

    # ─── BandShift paired t-test ───
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS — BandShift vs Pooled (paired, Bonferroni)")
    print("=" * 80)
    df_bs = df[(df['phase'] == 1) & (df['condition'] == 'Pooled+BandShift')]
    n_comp = 5 * len(CLASSIFIERS)  # 5 methods × 2 classifiers = 10
    ac = 0.05 / n_comp
    print(f"α_corrected = 0.05/{n_comp} = {ac:.4f}\n")
    print(f"{'Method':<10} {'Classifier':<14} {'t':>8} {'p_unc':>10} "
          f"{'p_corr':>10} {'Sig':>6}")
    print("-" * 60)
    for method_name, _, _ in METHODS:
        for clf in CLASSIFIERS:
            sc = df_pooled[(df_pooled['method'] == method_name) &
                           (df_pooled['classifier'] == clf)].sort_values('repeat')
            sb = df_bs[(df_bs['method'] == method_name) &
                       (df_bs['classifier'] == clf)].sort_values('repeat')
            if len(sc) == 0 or len(sb) == 0:
                continue
            d = sc.OA.values - sb.OA.values
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 1e-12 else 0.0
            pu = 2.0 * tdist.sf(abs(t), len(d) - 1) if se > 1e-12 else 1.0
            pc = min(pu * n_comp, 1.0)
            sig = "  ***" if pc < 0.05 else ""
            print(f"{method_name:<10} {clf_labels[clf]:<14} {t:>8.3f} "
                  f"{pu:>10.6f} {pc:>10.6f}{sig}")


if __name__ == "__main__":
    run()
