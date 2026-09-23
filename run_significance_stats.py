#!/usr/bin/env python3
"""
run_significance_stats.py
=========================
Compute all significance tests from existing raw data and output a single
structured CSV for the paper's significance table.

Inputs (no re-running of experiments needed):
  - results_baselines_kfold_raw.csv   (original plastic dataset, 5-fold CV x 5 seeds)
  - results_generalization_raw.csv    (generalization dataset, 5-fold CV x 5 seeds)

Only LinearSVM + RBF-SVM (RF excluded).

Tests computed (per dataset x classifier):
  1. RM-ANOVA         — overall method effect on Normal OA (F, p, df)
  2. Bonferroni posthoc — all C(5,2)=10 method pairs (t, p_unc, p_corr, sig)
  3. BandShift paired t — Normal vs BandShift per method (t, p_unc, p_corr, sig)

Output:
  results_significance_stats.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betainc
from scipy.stats import t as tdist

BASE_DIR = Path(__file__).parent
METHODS = ["USP", "SNV", "SG1", "EMSC", "Detrend"]
CLASSIFIERS = ["svm", "rbf_svm"]
CLF_LABELS = {"svm": "LinearSVM", "rbf_svm": "RBF-SVM"}
METRICS = ["OA", "macro_F1"]
ALPHA = 0.05


# ───────────────────────── Stats ─────────────────────────

def rm_anova(mat):
    """Repeated-measures ANOVA. mat: (n_subjects, k_treatments)."""
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


def posthoc_bonferroni(mat, names, alpha=ALPHA):
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


def paired_t(x, y, n_corr=5, alpha=ALPHA):
    """Paired t-test (x vs y), Bonferroni-corrected over n_corr comparisons."""
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se > 1e-12 else 0.0
    pu = 2.0 * tdist.sf(abs(t), n - 1) if se > 1e-12 else 1.0
    pc = min(pu * n_corr, 1.0)
    ac = alpha / n_corr
    return t, pu, pc, ac, pc < ac


# ───────────────────────── Per-dataset computation ─────────────────────────

def compute_dataset(df, dataset_name, normal_cond, bs_cond, index_cols):
    """Compute RM-ANOVA + Bonferroni + BandShift paired t for one dataset.

    df          : raw dataframe with columns [index_cols..., 'method',
                  'classifier', 'condition', 'OA', 'macro_F1']
    normal_cond : condition value for Normal OA
    bs_cond     : condition value for BandShift OA
    index_cols  : columns uniquely identifying one evaluation (e.g. ['seed','fold'])

    Tests are computed for BOTH metrics ('OA' and 'macro_F1'); the output
    carries a 'metric' column so tables can filter by metric.
    """
    rows = []
    for clf in CLASSIFIERS:
        sub = df[df['classifier'] == clf]
        dn = sub[sub['condition'] == normal_cond].copy()
        db = sub[sub['condition'] == bs_cond].copy()

        # align on index_cols
        dn = dn.sort_values(index_cols + ['method'])
        db = db.sort_values(index_cols + ['method'])

        for metric in METRICS:
            # ── RM-ANOVA on Normal ──
            piv = dn.pivot_table(index=index_cols, columns='method',
                                 values=metric)[METHODS]
            mat = piv.values
            F, p, dft, dfe = rm_anova(mat)
            rows.append({
                'dataset': dataset_name, 'classifier': CLF_LABELS[clf],
                'metric': metric,
                'test': 'RM-ANOVA', 'comparison': 'All methods',
                'stat_name': 'F', 'stat_value': round(F, 4),
                'df': f"({dft},{dfe})",
                'p_uncorrected': f"{p:.6g}",
                'p_corrected': f"{p:.6g}",
                'alpha_corrected': f"{ALPHA:.4f}",
                'significant': 'yes' if p < ALPHA else 'no',
            })

            # ── Bonferroni post-hoc ──
            ph, ac = posthoc_bonferroni(mat, METHODS, alpha=ALPHA)
            for pair, md, t, pu, pc, s in ph:
                rows.append({
                    'dataset': dataset_name, 'classifier': CLF_LABELS[clf],
                    'metric': metric,
                    'test': 'Bonferroni posthoc', 'comparison': pair,
                    'stat_name': 't', 'stat_value': round(t, 3),
                    'df': f"{len(mat)-1}",
                    'p_uncorrected': f"{pu:.6g}",
                    'p_corrected': f"{pc:.6g}",
                    'alpha_corrected': f"{ac:.4f}",
                    'significant': 'yes' if s else 'no',
                    'delta_pp': round(md * 100, 2),
                })

            # ── BandShift vs Normal paired t (per method) ──
            n_corr = len(METHODS)
            for method in METHODS:
                n = dn[dn['method'] == method].sort_values(index_cols)[metric].values
                b = db[db['method'] == method].sort_values(index_cols)[metric].values
                t, pu, pc, acm, s = paired_t(n, b, n_corr=n_corr, alpha=ALPHA)
                delta = (b.mean() - n.mean()) * 100
                rows.append({
                    'dataset': dataset_name, 'classifier': CLF_LABELS[clf],
                    'metric': metric,
                    'test': 'BandShift paired t', 'comparison': method,
                    'stat_name': 't', 'stat_value': round(t, 3),
                    'df': f"{len(n)-1}",
                    'p_uncorrected': f"{pu:.6g}",
                    'p_corrected': f"{pc:.6g}",
                    'alpha_corrected': f"{acm:.4f}",
                    'significant': 'yes' if s else 'no',
                    'delta_pp': round(delta, 2),
                })
    return rows


def print_summary(df_stats):
    """Print human-readable summary tables."""
    for (dataset, clf, metric), grp in df_stats.groupby(
            ['dataset', 'classifier', 'metric']):
        print("\n" + "=" * 92)
        print(f"{dataset}  |  {clf}  |  metric={metric}")
        print("=" * 92)

        # RM-ANOVA
        rma = grp[grp['test'] == 'RM-ANOVA'].iloc[0]
        print(f"[RM-ANOVA]  F{rma['df']}={rma['stat_value']}  "
              f"p={rma['p_uncorrected']}  "
              f"{'*** SIG' if rma['significant']=='yes' else 'n.s.'}")

        # Bonferroni
        ph = grp[grp['test'] == 'Bonferroni posthoc']
        ac = ph['alpha_corrected'].iloc[0]
        print(f"\n  Bonferroni posthoc (α_corr={ac}):")
        print(f"  {'Pair':<22} {'Δ(pp)':>9} {'t':>9} {'p_unc':>11} "
              f"{'p_corr':>11} {'Sig':>5}")
        print(f"  {'-'*67}")
        for _, r in ph.iterrows():
            sig = "***" if r['significant'] == 'yes' else ""
            dpp = r.get('delta_pp', '')
            dpp_s = f"{dpp:>8.2f}" if dpp != '' else f"{'':>8}"
            print(f"  {r['comparison']:<22} {dpp_s} {r['stat_value']:>9.3f} "
                  f"{r['p_uncorrected']:>11} {r['p_corrected']:>11} {sig:>5}")

        # BandShift paired t
        bt = grp[grp['test'] == 'BandShift paired t']
        ac = bt['alpha_corrected'].iloc[0]
        print(f"\n  BandShift vs Normal paired t (α_corr={ac}):")
        print(f"  {'Method':<12} {'Δ(pp)':>9} {'t':>9} {'p_unc':>11} "
              f"{'p_corr':>11} {'Sig':>5}")
        print(f"  {'-'*57}")
        for _, r in bt.iterrows():
            sig = "***" if r['significant'] == 'yes' else ""
            dpp = r.get('delta_pp', '')
            dpp_s = f"{dpp:>8.2f}" if dpp != '' else f"{'':>8}"
            print(f"  {r['comparison']:<12} {dpp_s} {r['stat_value']:>9.3f} "
                  f"{r['p_uncorrected']:>11} {r['p_corrected']:>11} {sig:>5}")


# ───────────────────────── Main ─────────────────────────

def run():
    all_rows = []

    # ── Original dataset ──
    orig = pd.read_csv(BASE_DIR / "results_baselines_kfold_raw.csv")
    orig = orig[(orig['ratio'] == 1.0) & (orig['method'].isin(METHODS))]
    print(f"[Original] {len(orig)} rows | classifiers: "
          f"{sorted(orig['classifier'].unique())}")
    all_rows += compute_dataset(
        orig, "Original (Clear water)",
        normal_cond='Normal', bs_cond='BandShift',
        index_cols=['seed', 'fold'])

    # ── Generalization dataset ──
    gen = pd.read_csv(BASE_DIR / "results_generalization_raw.csv")
    gen = gen[(gen['phase'] == 1) & (gen['method'].isin(METHODS))]
    print(f"[Generalization] {len(gen)} rows | classifiers: "
          f"{sorted(gen['classifier'].unique())}")
    all_rows += compute_dataset(
        gen, "Generalization (Pooled)",
        normal_cond='Pooled', bs_cond='Pooled+BandShift',
        index_cols=['seed', 'fold'])

    # ── Save CSV ──
    df_stats = pd.DataFrame(all_rows)
    out = BASE_DIR / "results_significance_stats.csv"
    df_stats.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df_stats)} rows)")

    # ── Print summary ──
    print_summary(df_stats)


if __name__ == "__main__":
    run()
