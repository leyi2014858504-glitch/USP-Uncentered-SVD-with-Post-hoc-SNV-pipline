#!/usr/bin/env python3
"""
run_latency_benchmark.py
========================
Computational latency benchmark: USP vs traditional preprocessing baselines.

Reviewer request: "discuss computational latency per sample would inform
the feasibility of industrial deployment."

Measures, for each of the 5 fair methods (LinearSVM, ratio=1.0 setup):
  1. Analytical FLOPs per test sample (preprocessing + projection)
  2. Wall-clock training time  (fit on the 4:1 training pool, per repeat)
  3. Wall-clock inference latency per test sample (preprocess + project +
     scale + SVM decision), median over many repeats

Setup matches the main experiment:
  - Original clear-plastic dataset, 940-1680 nm (741 bands), N_DIM=20
  - Single stratified 4:1 split (seed=42) for timing; timing is protocol-
    independent, so a single split suffices.

Outputs:
  results_latency_benchmark.csv
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_baselines_kfold import (DATA_DIR, N_DIM, snv, sg_first_derivative,
                                 emsc, detrend_only, bandshift)
from run_log_preprocessing import load_data

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
SPLIT_SEED = 42
N_REPEAT_FIT = 10       # training-time repeats
N_REPEAT_INFER = 200    # inference-time repeats
N_INFER_COPIES = 20     # batch size for inference timing


# ───────────────────────── Pipelines (mirroring run_baselines_kfold) ─────

def fit_pipeline(Xtr_pre, ytr, variant):
    """Fit once; return dict of fitted components (training path only)."""
    t0 = time.perf_counter()
    if variant == "A_postSNV":
        svd = TruncatedSVD(n_components=N_DIM, random_state=42)
        Ztr = svd.fit_transform(Xtr_pre)
        Ztr = snv(Ztr)
        sc = StandardScaler().fit(Ztr)
        clf = SVC(kernel='linear', C=1.0, class_weight='balanced',
                  random_state=42).fit(sc.transform(Ztr), ytr)
        state = dict(variant=variant, svd=svd, sc=sc, clf=clf, mu=None)
    else:
        mu = Xtr_pre.mean(axis=0)
        svd = TruncatedSVD(n_components=N_DIM, random_state=42)
        Ztr = svd.fit_transform(Xtr_pre - mu)
        sc = StandardScaler().fit(Ztr)
        clf = SVC(kernel='linear', C=1.0, class_weight='balanced',
                  random_state=42).fit(sc.transform(Ztr), ytr)
        state = dict(variant=variant, svd=svd, sc=sc, clf=clf, mu=mu)
    return state, time.perf_counter() - t0


def infer_batch(state, Xte_pre):
    """Per-batch inference: preprocess (done outside) → project → scale → predict."""
    if state["variant"] == "A_postSNV":
        Z = state["svd"].transform(Xte_pre)
        Z = snv(Z)
    else:
        Z = state["svd"].transform(Xte_pre - state["mu"])
    return state["clf"].predict(state["sc"].transform(Z))


# ───────────────────────── Analytical FLOPs (per test sample) ────────────

def flops_table(n_bands, k=N_DIM, n_classes=6):
    """Analytical multiply-add FLOPs per test sample (SVM handled separately)."""
    rows = []
    # shared: SVD projection  X(1×B) @ Vt(B×K)
    proj = 2 * n_bands * k
    rows.append(("SVD projection (shared)", proj))
    # USP: post-SNV in PC space (K dims) + scaler
    usp_post = 4 * k + 2 * k            # mean/std + sub/div, then scaler
    rows.append(("USP post-SNV in PC space", usp_post))
    # SNV-pre: SNV on full spectrum (B dims) + centering + scaler
    snv_full = 4 * n_bands + 2 * n_bands
    rows.append(("SNV-pre on full spectrum", snv_full))
    return rows, proj, usp_post, snv_full


def svm_decision_flops(n_sv_total, k=N_DIM, n_classes=6):
    """SVM decision FLOPs per test sample, both deployment forms:
    - primal: w precomputed once after training; OvO = n_cls(n_cls-1)/2
      classifiers x k dims. Identical for all methods (shared).
    - dual: libsvm predict() expands over support vectors; each class-pair
      classifier uses the SVs of its two classes, so the total kernel
      evaluations over all pairs = (n_classes - 1) * n_sv_total."""
    primal = 2 * (n_classes * (n_classes - 1) // 2) * k
    dual = 2 * (n_classes - 1) * n_sv_total * k
    return primal, dual


# ───────────────────────── Main ─────────────────────────

def main():
    csv_path = DATA_DIR / "spectradictionary.csv"
    X, y = load_data(str(csv_path), water_filter=0)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    n_bands = X.shape[1]

    # inference batch: replicate test set to reduce timer noise
    Xte_batch = np.tile(Xte, (N_INFER_COPIES, 1))

    PRE = {
        "USP":      lambda tr, te: (tr, te),
        "SNV":      lambda tr, te: (snv(tr), snv(te)),
        "SG1":      lambda tr, te: (sg_first_derivative(tr), sg_first_derivative(te)),
        "EMSC":     lambda tr, te: emsc(tr, te),
        "Detrend":  lambda tr, te: detrend_only(tr, te),
    }
    VARIANTS = {"USP": "A_postSNV", "SNV": "A2_noPostSNV", "SG1": "A2_noPostSNV",
                "EMSC": "A2_noPostSNV", "Detrend": "A2_noPostSNV"}

    rows = []
    print("=" * 92)
    print(f"Latency benchmark | {n_bands} bands | k={N_DIM} PCs | "
          f"train={len(ytr)} test={len(yte)} | LinearSVM")
    print("=" * 92)

    # analytical FLOPs (preprocessing differences; SVD projection shared;
    # SVM decision reported separately per method: primal form is shared,
    # dual form scales with the measured number of support vectors)
    fl_rows, proj, usp_post, snv_full = flops_table(n_bands)
    print("\nAnalytical FLOPs per test sample (multiply-adds):")
    for name, v in fl_rows:
        print(f"  {name:<28} {v:>10,}")
    svm_primal, _ = svm_decision_flops(0)
    usp_total = proj + usp_post + svm_primal
    snv_total = proj + snv_full + svm_primal
    print(f"  {'SVM decision, primal (shared)':<28} {svm_primal:>10,}")
    print(f"  {'USP total (primal)':<28} {usp_total:>10,}")
    print(f"  {'SNV-pre total (primal)':<28} {snv_total:>10,}  "
          f"(+{(snv_total - usp_total) / usp_total * 100:.1f}%)")

    for name, pp in PRE.items():
        Xtr_pp, Xte_pp = pp(Xtr, Xte)
        Xte_pp_batch = pp(Xtr, Xte_batch)[1]

        # ── training time (includes preprocessing of train set) ──
        fit_times = []
        for _ in range(N_REPEAT_FIT):
            state, t_fit = fit_pipeline(Xtr_pp, ytr, VARIANTS[name])
            fit_times.append(t_fit)
        fit_ms_total = np.median(fit_times) * 1000
        fit_ms_per_train = fit_ms_total / len(ytr)
        # measured support vectors -> dual-form decision cost
        n_sv_total = int(np.asarray(state["clf"].n_support_).sum())
        svm_primal_m, svm_dual_m = svm_decision_flops(n_sv_total)

        # ── inference latency per sample (preprocess+project+scale+predict) ──
        infer_times = []
        for _ in range(N_REPEAT_INFER):
            t0 = time.perf_counter()
            Xb_pp = pp(Xtr, Xte_batch)[1]        # per-batch preprocessing
            infer_batch(state, Xb_pp)
            infer_times.append(time.perf_counter() - t0)
        # subtract the timing of the no-op preprocessing call for "none"
        t_infer = np.median(infer_times) / len(Xte_batch) * 1e6  # µs/sample

        # pure model latency (projection+scale+predict only, no spectral prep)
        pure_times = []
        for _ in range(N_REPEAT_INFER):
            t0 = time.perf_counter()
            infer_batch(state, Xte_pp_batch)
            pure_times.append(time.perf_counter() - t0)
        t_pure = np.median(pure_times) / len(Xte_batch) * 1e6

        rows.append({
            "method": name,
            "fit_total_ms": fit_ms_total,
            "fit_ms_per_train_sample": fit_ms_per_train,
            "infer_us_per_sample_full": t_infer,   # incl. spectral preprocessing
            "infer_us_per_sample_model": t_pure,   # projection+scale+SVM only
            "n_sv_total": n_sv_total,
            "svm_flops_primal": svm_primal_m,      # shared (w precomputed)
            "svm_flops_dual": svm_dual_m,          # libsvm SV expansion
        })
        print(f"\n{name}:  fit {fit_ms_total:7.1f} ms total "
              f"({fit_ms_per_train:5.2f} ms/train-sample) | "
              f"infer {t_infer:8.1f} µs/sample full | "
              f"{t_pure:8.1f} µs/sample model-only | "
              f"n_sv={n_sv_total} (dual {svm_dual_m:,} FLOPs)")

    df = pd.DataFrame(rows)
    out = BASE_DIR / "results_latency_benchmark.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved: {out}")

    # relative comparison table
    base = df[df["method"] == "USP"].iloc[0]
    print("\n" + "=" * 92)
    print("Relative cost (USP = 1.00)")
    print("=" * 92)
    print(f"{'method':<10}{'fit total':>12}{'fit/sample':>12}"
          f"{'infer full':>14}{'infer model':>14}")
    for _, r in df.iterrows():
        print(f"{r['method']:<10}{r['fit_total_ms'] / base['fit_total_ms']:>11.2f}x"
              f"{r['fit_ms_per_train_sample'] / base['fit_ms_per_train_sample']:>11.2f}x"
              f"{r['infer_us_per_sample_full'] / base['infer_us_per_sample_full']:>13.2f}x"
              f"{r['infer_us_per_sample_model'] / base['infer_us_per_sample_model']:>13.2f}x")


if __name__ == "__main__":
    main()
