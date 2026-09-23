#!/usr/bin/env python3
"""
run_log_preprocessing.py
========================
Test whether log-transforming raw spectra (X = ln(X + eps)) before USP
improves classification. Motivation: Mie scattering follows a power law
I ∝ λ^(-n); taking log linearizes this multiplicative structure, which
may help TruncSVD capture scatter intensity more cleanly in PC1.

Compares 3 methods per water condition:
  A      (USP)        : Raw -> Uncentered SVD -> SNV(PC) -> Scale -> SVM
  A_log  (log+USP)    : log(X+eps) -> Uncentered SVD -> SNV(PC) -> Scale -> SVM
  A2     (SNV-Pre)    : SNV -> Centered SVD -> Scale -> SVM   (reference baseline)

Protocol (current, consistent with run_baselines_kfold.py):
  - 5-fold stratified CV x 5 seeds [42,43,44,45,46] = 25 evaluations
  - LinearSVM (C=1.0, class_weight='balanced')
  - 3 water conditions: Clear / Turbid / Foamy
  - Reuses pipeline_usp / pipeline_traditional from run_baselines_kfold
    (correct scaler fit-on-train; the old version re-fit StandardScaler on
    the test set, a leakage bug — now fixed by sharing the verified pipeline).

Significance: per condition, RM-ANOVA + Bonferroni post-hoc on the 3 methods,
for OA and macro_F1.

Outputs:
  results_log_preprocessing_raw.csv     — per seed/fold (for stats)
  results_log_preprocessing.csv         — aggregated summary
  results_log_preprocessing_stats.csv   — RM-ANOVA + Bonferroni
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold

from run_baselines_kfold import (
    pipeline_usp, pipeline_traditional, snv, evaluate,
    rm_anova, posthoc_bonferroni,
    SEEDS, N_SPLITS, N_DIM, DATA_DIR,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
LOG_EPS = 1e-4
ALPHA = 0.05

METHOD_NAMES = ["A (USP)", "A_log (log+USP)", "A2 (SNV-Pre)"]
CLF = "svm"                       # LinearSVM only (faithful to original scope)
CLF_LABEL = "LinearSVM"


# ───────────────────────── Data ─────────────────────────

def load_data(csv_path, water_filter=None):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["code", "fraction"] + list(range(350, 2501))
    df["material"] = (df["code"] // 10).astype(int)
    df["water"] = (df["code"] % 10).astype(int)
    df = df[df["material"] != 6]           # exclude Mix
    if water_filter is not None:
        df = df[df["water"] == water_filter]

    band_cols = list(range(940, 1681))
    X = df[band_cols].values.astype(np.float32)
    y = df["material"].values.astype(np.int32)

    names = {1: "PET", 2: "HDPE", 3: "LDPE", 4: "PP", 5: "EPSF", 7: "Weathered"}
    print(f"[Data] water={water_filter} | {len(y)} samples | "
          f"{len(np.unique(y))} classes | {X.shape[1]} bands")
    for c in np.unique(y):
        print(f"  {names.get(int(c), c)}: {(y == c).sum()}")
    return X, y


def log_transform(X):
    return np.log(X + LOG_EPS).astype(np.float32)


# ───────────────────────── Method runners ─────────────────────────
# Each returns y_pred. A and A2 reuse the verified baselines_kfold pipelines
# (correct fit-on-train StandardScaler + n_components clamp); A_log is USP
# applied to log-transformed input.

def run_A(Xtr, Xte, ytr):
    y_pred, *_ = pipeline_usp(Xtr, Xte, ytr, CLF)
    return y_pred


def run_A_log(Xtr, Xte, ytr):
    y_pred, *_ = pipeline_usp(log_transform(Xtr), log_transform(Xte), ytr, CLF)
    return y_pred


def run_A2(Xtr, Xte, ytr):
    # SNV-Pre baseline: SNV -> centered SVD (pipeline_traditional expects
    # preprocessed input).
    y_pred, *_ = pipeline_traditional(snv(Xtr), snv(Xte), ytr, CLF)
    return y_pred


PIPES = [(METHOD_NAMES[0], run_A),
         (METHOD_NAMES[1], run_A_log),
         (METHOD_NAMES[2], run_A2)]


# ───────────────────────── Per-condition experiment ─────────────────────────

def run_condition(X, y, label, raw_records):
    """5-fold CV x 5 seeds for the 3 methods; appends per-fold-seed rows."""
    n_eval = 0
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr].astype(np.float32), X[te].astype(np.float32)
            ytr, yte = y[tr], y[te]
            for mname, fn in PIPES:
                y_pred = fn(Xtr, Xte, ytr)
                m = evaluate(yte, y_pred)
                raw_records.append({
                    "condition": label, "seed": seed, "fold": fold,
                    "method": mname, "classifier": CLF, **m,
                })
                n_eval += 1
    print(f"  [{label}] {n_eval} evals done")
    return raw_records


# ───────────────────────── Stats ─────────────────────────

def compute_stats(raw, condition, rows):
    """RM-ANOVA + Bonferroni post-hoc on 3 methods, for OA and macro_F1."""
    for metric in ["OA", "macro_F1"]:
        piv = raw.pivot_table(index=["seed", "fold"], columns="method",
                              values=metric)[METHOD_NAMES]
        mat = piv.values
        F, p, dft, dfe = rm_anova(mat)
        rows.append({
            "dataset": condition, "classifier": CLF_LABEL, "metric": metric,
            "test": "RM-ANOVA", "comparison": "All methods",
            "stat_name": "F", "stat_value": round(F, 4),
            "df": f"({dft},{dfe})",
            "p_uncorrected": f"{p:.6g}", "p_corrected": f"{p:.6g}",
            "alpha_corrected": f"{ALPHA:.4f}",
            "significant": "yes" if p < ALPHA else "no",
        })
        ph, ac = posthoc_bonferroni(mat, METHOD_NAMES, alpha=ALPHA)
        for pair, md, t, pu, pc, s in ph:
            rows.append({
                "dataset": condition, "classifier": CLF_LABEL, "metric": metric,
                "test": "Bonferroni posthoc", "comparison": pair,
                "stat_name": "t", "stat_value": round(t, 3),
                "df": f"{len(mat) - 1}",
                "p_uncorrected": f"{pu:.6g}", "p_corrected": f"{pc:.6g}",
                "alpha_corrected": f"{ac:.4f}",
                "significant": "yes" if s else "no",
                "delta_pp": round(md * 100, 2),
            })
    return rows


# ───────────────────────── Main ─────────────────────────

def run():
    csv_path = DATA_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        csv_path = BASE_DIR / "spectradictionary.csv"
    if not csv_path.exists():
        print(f"ERROR: spectradictionary.csv not found")
        return

    water_conditions = [("Clear", 0), ("Turbid", 1), ("Foamy", 2)]
    print(f"Protocol: 5-fold CV x {len(SEEDS)} seeds = "
          f"{N_SPLITS * len(SEEDS)} evals/method/condition")
    print(f"Methods: {METHOD_NAMES}")
    print(f"Classifier: {CLF_LABEL} only")

    raw_records = []
    for wlabel, wfilter in water_conditions:
        X, y = load_data(csv_path, water_filter=wfilter)
        run_condition(X, y, wlabel, raw_records)

        # PC1 energy check (log vs original)
        X_f = X.astype(np.float32)
        svd_o = TruncatedSVD(n_components=N_DIM, random_state=42).fit(X_f)
        pc1_o = (svd_o.singular_values_[0] ** 2) / np.sum(X_f ** 2) * 100
        X_l = log_transform(X_f)
        svd_l = TruncatedSVD(n_components=N_DIM, random_state=42).fit(X_l)
        pc1_l = (svd_l.singular_values_[0] ** 2) / np.sum(X_l ** 2) * 100
        print(f"  PC1 energy ({wlabel}): Original={pc1_o:.2f}%  "
              f"Log-transformed={pc1_l:.2f}%")

    # ─── Raw + summary ───
    df_raw = pd.DataFrame(raw_records)
    out_raw = BASE_DIR / "results_log_preprocessing_raw.csv"
    df_raw.to_csv(out_raw, index=False, float_format="%.6f")
    print(f"\nSaved raw: {out_raw} ({len(df_raw)} rows)")

    summary = df_raw.groupby(["condition", "method"]).agg(
        OA_mean=("OA", "mean"), OA_std=("OA", "std"),
        F1_mean=("macro_F1", "mean"), F1_std=("macro_F1", "std"),
        kappa_mean=("kappa", "mean"), kappa_std=("kappa", "std"),
    ).reset_index()
    order = {n: i for i, n in enumerate(METHOD_NAMES)}
    summary["_o"] = summary["method"].map(order)
    summary = summary.sort_values(["condition", "_o"]).drop(columns=["_o"])
    out_sum = BASE_DIR / "results_log_preprocessing.csv"
    summary.to_csv(out_sum, index=False, float_format="%.4f")
    print(f"Saved summary: {out_sum}")

    # ─── Significance stats ───
    stat_rows = []
    for wlabel, _ in water_conditions:
        cond_raw = df_raw[df_raw["condition"] == wlabel]
        compute_stats(cond_raw, wlabel, stat_rows)
    df_stats = pd.DataFrame(stat_rows)
    out_stats = BASE_DIR / "results_log_preprocessing_stats.csv"
    df_stats.to_csv(out_stats, index=False, float_format="%.6f")
    print(f"Saved stats: {out_stats} ({len(df_stats)} rows)")

    # ─── Print summary ───
    print("\n" + "=" * 92)
    print("Log-preprocessing results (5-fold CV x 5 seeds, LinearSVM)")
    print("=" * 92)
    for wlabel, _ in water_conditions:
        print(f"\n[{wlabel}]")
        print(f"  {'Method':<20} {'OA':>16} {'Macro-F1':>16} {'Kappa':>14}")
        sub = summary[summary["condition"] == wlabel]
        for _, r in sub.iterrows():
            oa = f"{r['OA_mean']*100:.2f} ± {r['OA_std']*100:.2f}"
            f1 = f"{r['F1_mean']*100:.2f} ± {r['F1_std']*100:.2f}"
            kp = f"{r['kappa_mean']:.4f} ± {r['kappa_std']:.4f}"
            print(f"  {r['method']:<20} {oa:>16} {f1:>16} {kp:>14}")

    print("\n" + "=" * 92)
    print("Significance (RM-ANOVA + Bonferroni post-hoc, α=0.05)")
    print("=" * 92)
    for (cond, metric), grp in df_stats.groupby(["dataset", "metric"]):
        rma = grp[grp["test"] == "RM-ANOVA"]
        if len(rma):
            r = rma.iloc[0]
            print(f"\n  [{cond} | {metric}]  RM-ANOVA F{r['df']}={r['stat_value']}  "
                  f"p={r['p_uncorrected']}  "
                  f"{'*** SIG' if r['significant']=='yes' else 'n.s.'}")
        ph = grp[grp["test"] == "Bonferroni posthoc"]
        if len(ph):
            ac = ph["alpha_corrected"].iloc[0]
            print(f"    Bonferroni post-hoc (α_corr={ac}):")
            for _, r in ph.iterrows():
                sig = " ***" if r["significant"] == "yes" else ""
                dpp = r.get("delta_pp", "")
                dpp_s = f"{dpp:>7.2f}" if dpp != "" else f"{'':>7}"
                print(f"      {r['comparison']:<26} Δ={dpp_s}pp  "
                      f"t={r['stat_value']:.3f}  p_corr={r['p_corrected']}{sig}")


if __name__ == "__main__":
    run()
