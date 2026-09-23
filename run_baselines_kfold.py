#!/usr/bin/env python3
"""
run_baselines_kfold.py
======================
Add 3 new baselines (SG1, EMSC, Detrend) to the existing learning-curve table
using the SAME protocol as usp_pipeline.py:

  - Data: spectradictionary.csv, Clear water only, 193 samples, 941-1680 nm
  - CV:   StratifiedKFold(n_splits=5, shuffle=True) x 5 seeds [42,43,44,45,46]
  - Ratios: [0.10, 0.30, 1.00]  (user-specified subset of original 7 ratios)
  - Classifier: LinearSVM (SVC kernel='linear', C=1.0, class_weight='balanced')
  - BandShift: at ratio=1.0, np.roll(X, shift=2, axis=1) on test set

Five methods (fair, same-level comparison — no chaining):
  1. USP      (proposed)  : Raw -> Uncentered TruncSVD(20) -> SNV(PC) -> Scale -> SVM
  2. SNV      (traditional): SNV(Raw) -> Centered TruncSVD(20) -> Scale -> SVM
  3. SG1      (traditional): SG1(Raw) -> Centered TruncSVD(20) -> Scale -> SVM
  4. EMSC     (traditional): EMSC(Raw) -> Centered TruncSVD(20) -> Scale -> SVM
  5. Detrend  (traditional): Detrend(Raw) -> Centered TruncSVD(20) -> Scale -> SVM

Outputs:
  results_baselines_kfold.csv          — learning-curve summary (appendable to csv_results_summary.csv)
  results_baselines_kfold_bandshift.csv — BandShift summary at ratio=1.0
  results_baselines_kfold_raw.csv      — per-fold-seed raw data (for RM-ANOVA)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.signal import savgol_filter
from scipy.special import betainc
from scipy.stats import t as tdist

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "1"
N_DIM = 20
SEEDS = [42, 43, 44, 45, 46]
RATIOS = [0.10, 0.30, 1.00]
N_SPLITS = 5


# ───────────────────────── Data ─────────────────────────

def load_clear_plastic(csv_path, band_range=(940, 1680)):
    """Load Clear-water river plastics (same as usp_pipeline.load_river_csv)."""
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["code", "fraction"] + list(range(350, 2501))
    df["material"] = (df["code"] // 10).astype(int)
    df["water"] = (df["code"] % 10).astype(int)
    df = df[df["material"] != 6]            # exclude Mix
    df = df[df["water"] == 0]               # Clear only

    band_cols = list(range(band_range[0], band_range[1] + 1))
    X = df[band_cols].values.astype(np.float32)
    y = df["material"].values.astype(np.int32)
    return X, y


# ───────────────────────── Preprocessing ─────────────────────────

def snv(X, eps=1e-8):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def sg_first_derivative(X, window=15, polyorder=2):
    return savgol_filter(X, window_length=window, polyorder=polyorder,
                         deriv=1, delta=1.0, axis=1).astype(np.float32)


def emsc(X_train, X_test, poly_order=2):
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

    return _correct(X_train), _correct(X_test)


def detrend_only(X_train, X_test, poly_order=2):
    n_bands = X_train.shape[1]
    w = np.linspace(-1, 1, n_bands)
    A = np.vander(w, poly_order + 1, increasing=True)

    def _detrend(X):
        Xd = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            coefs, *_ = np.linalg.lstsq(A, X[i], rcond=None)
            Xd[i] = X[i] - A @ coefs
        return Xd.astype(np.float32)

    return _detrend(X_train), _detrend(X_test)


# ───────────────────────── Classifier ─────────────────────────

def make_clf(clf_type='svm', C=1.0, gamma='scale'):
    if clf_type == 'rbf_svm':
        return SVC(kernel='rbf', C=C, gamma=gamma,
                   class_weight='balanced', random_state=42)
    return SVC(kernel='linear', C=C, class_weight='balanced', random_state=42)


CLASSIFIERS = ["svm", "rbf_svm"]  # RF dropped
CLF_LABELS = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}


# ───────────────────────── Pipelines ─────────────────────────

def pipeline_usp(Xtr, Xte, ytr, clf_type='svm', C=1.0, gamma='scale'):
    """USP: Raw -> Uncentered TruncSVD(20) -> SNV(PC) -> Scale -> SVM."""
    # Clamp n_components to <= n_train-1 to avoid zero-singular-value PCs
    # (matches usp_pipeline._comp_pca_fit; prevents StandardScaler blow-up at
    #  low training ratios where n_train < N_DIM).
    n_comp = max(1, min(N_DIM, Xtr.shape[0] - 1, Xtr.shape[1]))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Ztr = svd.fit_transform(Xtr)
    Zte = svd.transform(Xte)
    Ztr = snv(Ztr)
    Zte = snv(Zte)
    sc = StandardScaler().fit(Ztr)
    clf = make_clf(clf_type, C=C, gamma=gamma)
    clf.fit(sc.transform(Ztr), ytr)
    y_pred = clf.predict(sc.transform(Zte))
    return y_pred, svd, clf, sc


def pipeline_traditional(Xtr_pp, Xte_pp, ytr, clf_type='svm', C=1.0, gamma='scale'):
    """Traditional: preprocessed -> Centered TruncSVD(20) -> Scale -> SVM."""
    mu = Xtr_pp.mean(axis=0)
    # Clamp n_components to <= n_train-1 (centering drops rank by 1, so the
    # n_train-th PC has singular value 0; its arbitrary Vt direction creates a
    # garbage feature on the test set that StandardScaler passes through).
    n_comp = max(1, min(N_DIM, Xtr_pp.shape[0] - 1, Xtr_pp.shape[1]))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Ztr = svd.fit_transform(Xtr_pp - mu)
    Zte = svd.transform(Xte_pp - mu)
    sc = StandardScaler().fit(Ztr)
    clf = make_clf(clf_type, C=C, gamma=gamma)
    clf.fit(sc.transform(Ztr), ytr)
    y_pred = clf.predict(sc.transform(Zte))
    return y_pred, svd, clf, sc, mu


def apply_bandshift_usp(Xte_pp_bs, svd, sc):
    Zte = svd.transform(Xte_pp_bs)
    Zte = snv(Zte)
    return sc.transform(Zte)


def apply_bandshift_traditional(Xte_pp_bs, svd, sc, mu):
    Zte = svd.transform(Xte_pp_bs - mu)
    return sc.transform(Zte)


# ───────────────────────── Eval ─────────────────────────

def evaluate(y_true, y_pred):
    return {
        'OA': float(accuracy_score(y_true, y_pred)),
        'macro_F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
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


# ───────────────────────── Main ─────────────────────────

METHOD_NAMES = ["USP", "SNV", "SG1", "EMSC", "Detrend"]


def run():
    csv_path = DATA_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        csv_path = BASE_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        print(f"ERROR: spectradictionary.csv not found")
        return

    X, y = load_clear_plastic(csv_path)
    n_cls = len(np.unique(y))
    print(f"[Data] {len(y)} samples | {n_cls} classes | {X.shape[1]} bands (940-1680 nm)")
    for c in np.unique(y):
        print(f"  class {int(c)}: {(y == c).sum()} samples")

    print(f"\nProtocol: 5-fold CV x {len(SEEDS)} seeds = {N_SPLITS * len(SEEDS)} evaluations")
    print(f"Ratios: {RATIOS}")
    print(f"Methods: {METHOD_NAMES}")
    print(f"BandShift: at ratio=1.0, shift=2")

    records_lc = []      # learning curve
    records_bs = []      # bandshift (ratio=1.0)
    records_raw = []     # per-fold-seed raw (for stats)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_pool, y_pool = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_test = X_test.astype(np.float32)

            # BandShift test set (rolled before preprocessing)
            X_test_bs = bandshift(X_test, shift=2)

            for ratio in RATIOS:
                n_tr = max(int(ratio * len(y_pool)), n_cls)
                if n_tr >= len(y_pool):
                    Xtr, ytr = X_pool, y_pool
                else:
                    Xtr, _, ytr, _ = train_test_split(
                        X_pool, y_pool, train_size=n_tr,
                        stratify=y_pool, random_state=seed + 1000 * fold_idx)
                Xtr = Xtr.astype(np.float32)

                # Apply preprocessing (on training subset only — no leakage)
                Xtr_snv = snv(Xtr)
                Xte_snv = snv(X_test)
                Xtr_sg1 = sg_first_derivative(Xtr)
                Xte_sg1 = sg_first_derivative(X_test)
                Xtr_emsc, Xte_emsc = emsc(Xtr, X_test)
                Xtr_dt, Xte_dt = detrend_only(Xtr, X_test)

                # Also precompute BandShift preprocessed versions
                Xte_snv_bs = snv(X_test_bs)
                Xte_sg1_bs = sg_first_derivative(X_test_bs)
                _, Xte_emsc_bs = emsc(Xtr, X_test_bs)
                _, Xte_dt_bs = detrend_only(Xtr, X_test_bs)

                pp_map = {
                    "USP":     (Xtr,        X_test,      X_test_bs),
                    "SNV":     (Xtr_snv,    Xte_snv,     Xte_snv_bs),
                    "SG1":     (Xtr_sg1,    Xte_sg1,     Xte_sg1_bs),
                    "EMSC":    (Xtr_emsc,   Xte_emsc,    Xte_emsc_bs),
                    "Detrend": (Xtr_dt,     Xte_dt,      Xte_dt_bs),
                }

                for method in METHOD_NAMES:
                    Xtr_pp, Xte_pp, Xte_pp_bs = pp_map[method]

                    for clf_type in CLASSIFIERS:
                        # Normal
                        if method == "USP":
                            y_pred, svd, clf, sc = pipeline_usp(
                                Xtr_pp, Xte_pp, ytr, clf_type)
                            mu = None
                        else:
                            y_pred, svd, clf, sc, mu = pipeline_traditional(
                                Xtr_pp, Xte_pp, ytr, clf_type)
                        m = evaluate(y_test, y_pred)

                        records_lc.append({
                            'seed': seed, 'fold': fold_idx, 'ratio': ratio,
                            'method': method, 'classifier': clf_type, **m,
                        })
                        if ratio == 1.0:
                            records_raw.append({
                                'seed': seed, 'fold': fold_idx, 'ratio': ratio,
                                'method': method, 'classifier': clf_type,
                                'condition': 'Normal', **m,
                            })

                        # BandShift (only at ratio=1.0)
                        if ratio == 1.0:
                            if method == "USP":
                                Xte_final = apply_bandshift_usp(Xte_pp_bs, svd, sc)
                            else:
                                Xte_final = apply_bandshift_traditional(
                                    Xte_pp_bs, svd, sc, mu)
                            y_pred_bs = clf.predict(Xte_final)
                            m_bs = evaluate(y_test, y_pred_bs)
                            records_bs.append({
                                'seed': seed, 'fold': fold_idx, 'ratio': ratio,
                                'method': method, 'classifier': clf_type, **m_bs,
                            })
                            records_raw.append({
                                'seed': seed, 'fold': fold_idx, 'ratio': ratio,
                                'method': method, 'classifier': clf_type,
                                'condition': 'BandShift', **m_bs,
                            })

        print(f"  seed {seed} done")

    # ─── Save learning curve summary ───
    df_lc = pd.DataFrame(records_lc)
    summary_lc = df_lc.groupby(['ratio', 'method', 'classifier']).agg(
        OA_mean=('OA', 'mean'), OA_std=('OA', 'std'),
        macro_F1_mean=('macro_F1', 'mean'), macro_F1_std=('macro_F1', 'std'),
        kappa_mean=('kappa', 'mean'), kappa_std=('kappa', 'std'),
    ).reset_index()
    out_lc = BASE_DIR / "results_baselines_kfold.csv"
    summary_lc.to_csv(out_lc, index=False, float_format="%.4f")
    print(f"\nSaved learning curve: {out_lc}")

    # ─── Save BandShift summary ───
    df_bs = pd.DataFrame(records_bs)
    summary_bs = df_bs.groupby(['method', 'classifier']).agg(
        OA_mean=('OA', 'mean'), OA_std=('OA', 'std'),
        macro_F1_mean=('macro_F1', 'mean'), macro_F1_std=('macro_F1', 'std'),
        kappa_mean=('kappa', 'mean'), kappa_std=('kappa', 'std'),
    ).reset_index()
    out_bs = BASE_DIR / "results_baselines_kfold_bandshift.csv"
    summary_bs.to_csv(out_bs, index=False, float_format="%.4f")
    print(f"Saved BandShift: {out_bs}")

    # ─── Save raw ───
    df_raw = pd.DataFrame(records_raw)
    out_raw = BASE_DIR / "results_baselines_kfold_raw.csv"
    df_raw.to_csv(out_raw, index=False, float_format="%.6f")
    print(f"Saved raw: {out_raw}")

    # ─── Print learning curve table ───
    for clf_type in CLASSIFIERS:
        print("\n" + "=" * 90)
        print(f"Learning Curve (5-fold CV x 5 seeds, {CLF_LABELS[clf_type]}, Clear water)")
        print("=" * 90)
        print(f"{'Ratio':<8} {'Method':<12} {'OA':>18} {'Macro-F1':>18} {'Kappa':>14}")
        print("-" * 72)
        for ratio in RATIOS:
            for method in METHOD_NAMES:
                row = summary_lc[(summary_lc['ratio'] == ratio) &
                                 (summary_lc['method'] == method) &
                                 (summary_lc['classifier'] == clf_type)]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                oa = f"{r['OA_mean']*100:.2f} ± {r['OA_std']*100:.2f}"
                f1 = f"{r['macro_F1_mean']*100:.2f} ± {r['macro_F1_std']*100:.2f}"
                kp = f"{r['kappa_mean']:.4f} ± {r['kappa_std']:.4f}"
                print(f"{ratio:<8.2f} {method:<12} {oa:>18} {f1:>18} {kp:>14}")
            print()

    # ─── Print BandShift table + stats (per classifier) ───
    df_normal = df_raw[df_raw['condition'] == 'Normal']
    df_bs_raw = df_raw[df_raw['condition'] == 'BandShift']

    for clf_type in CLASSIFIERS:
        print("=" * 90)
        print(f"BandShift Robustness & Stats (ratio=1.0, {CLF_LABELS[clf_type]})")
        print("=" * 90)
        dn = df_normal[df_normal['classifier'] == clf_type]
        db = df_bs_raw[df_bs_raw['classifier'] == clf_type]
        print(f"{'Method':<12} {'OA(Normal)':>14} {'OA(BandShift)':>16} {'ΔOA(pp)':>10}")
        print("-" * 56)
        for method in METHOD_NAMES:
            n = dn[dn['method'] == method]['OA'].values
            b = db[db['method'] == method]['OA'].values
            delta = (b.mean() - n.mean()) * 100
            print(f"{method:<12} {n.mean()*100:>13.2f}% {b.mean()*100:>15.2f}% {delta:>9.2f}")

        # RM-ANOVA on Normal
        piv = dn.pivot_table(index=['seed', 'fold'],
                             columns='method', values='OA')[METHOD_NAMES]
        mat = piv.values
        F, p, dft, dfe = rm_anova(mat)
        print(f"\n[Normal] RM-ANOVA  F({dft},{dfe})={F:.4f}  p={p:.6f}"
              f"{'  *** SIG' if p < 0.05 else ''}")
        ph, ac = posthoc_bonferroni(mat, METHOD_NAMES, alpha=0.05)
        print(f"  Post-hoc (Bonferroni, α_corr={ac:.4f}):")
        print(f"  {'Pair':<24} {'Δ(OA)%':>8} {'t':>8} {'p_unc':>10} {'p_corr':>10} {'Sig':>6}")
        print(f"  {'-'*66}")
        for pair, md, t, pu, pc, s in ph:
            sig = "  ***" if s else ""
            print(f"  {pair:<24} {md*100:>7.2f}% {t:>8.3f} {pu:>10.6f} {pc:>10.6f}{sig}")

        # BandShift vs Normal paired t-test
        print(f"\n[BandShift vs Normal] paired t-test (Bonferroni α_corr={0.05/5:.4f})")
        print(f"  {'Method':<12} {'t':>8} {'p_unc':>10} {'p_corr':>10} {'Sig':>6}")
        print(f"  {'-'*48}")
        for method in METHOD_NAMES:
            n = dn[dn['method'] == method].sort_values(['seed', 'fold'])['OA'].values
            b = db[db['method'] == method].sort_values(['seed', 'fold'])['OA'].values
            d = n - b
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 1e-12 else 0.0
            pu = 2.0 * tdist.sf(abs(t), len(d) - 1) if se > 1e-12 else 1.0
            pc = min(pu * 5, 1.0)
            sig = "  ***" if pc < 0.05 else ""
            print(f"  {method:<12} {t:>8.3f} {pu:>10.6f} {pc:>10.6f}{sig}")
        print()


if __name__ == "__main__":
    run()
