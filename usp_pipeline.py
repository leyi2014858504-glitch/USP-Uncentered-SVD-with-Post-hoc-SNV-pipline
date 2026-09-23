#!/usr/bin/env python3
"""
usp_pipeline.py
===============
Reproducible pipeline for the paper:
  "Uncentered SVD + Post-SNV for SWIR Plastic Classification"

Methods:
  A.   Global TruncSVD (no centering)          — proposed
  A*.  Global TruncSVD (centered)               — ablation
  A2.  Global TruncSVD (SNV-Pre, no centering)  — ablation
  A2*. Global TruncSVD (SNV-Pre, centered)      — ablation
  A3.  Global TruncSVD (MSC-Pre, no centering)  — ablation
  A3*. Global TruncSVD (MSC-Pre, centered)      — ablation
  F.   PLS-DA (baseline)

Experiments:
  1. Stratified 5-fold learning curves (mean ± std over 5 seeds)
  2. Water OOD: ID=Clear, OOD={Turbid,Foamy} + simulated perturbations
  3. Loading vectors plot (PC1 & PC2)

Metrics: OA, AA, macro-F1, Cohen κ

Data:
  CSV reflectance library (350–2500 nm, cropped to 940–1680 nm)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, balanced_accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

N_DIM = 20

MATERIAL_NAMES = {
    1: "PET", 2: "HDPE", 3: "LDPE",
    4: "PP",  5: "EPSF", 7: "Weathered"
}
WATER_NAMES = {0: "Clear", 1: "Turbid", 2: "Foamy"}


def load_river_csv(csv_path,
                   min_plastic_frac=0.0,
                   exclude_mix=True,
                   water_filter=None,
                   band_range=(940, 1680)):
    """Load the Olyaei et al. (2024) river plastics CSV dataset."""
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["code", "fraction"] + list(range(350, 2501))

    df["material"] = (df["code"] // 10).astype(int)
    df["water"] = (df["code"] % 10).astype(int)

    df = df[df["fraction"] >= min_plastic_frac]
    if exclude_mix:
        df = df[df["material"] != 6]
    if water_filter is not None:
        df = df[df["water"] == water_filter]

    all_wvl = np.arange(350, 2501, dtype=float)
    band_mask = (all_wvl >= band_range[0]) & (all_wvl <= band_range[1])
    wvl = all_wvl[band_mask]
    band_cols = list(all_wvl[band_mask].astype(int))

    X = df[band_cols].values.astype(np.float32)
    y = df["material"].values.astype(np.int32)
    meta = df[["code", "material", "water", "fraction"]].reset_index(drop=True)

    print(f"[CSV] loaded {len(y)} samples | "
          f"{len(np.unique(y))} materials | "
          f"{X.shape[1]} bands ({band_range[0]}–{band_range[1]} nm)")
    for m in np.unique(y):
        print(f"  {MATERIAL_NAMES.get(int(m), int(m))}: {(y == m).sum()} samples")

    return X, y, wvl, meta


def run_csv_experiment_kfold(csv_path,
                             ratios=None,
                             n_splits=5,
                             seeds=None,
                             min_plastic_frac=0.0,
                             water_filter=0,
                             band_range=(940, 1680),
                             quiet=False):
    """Stratified k-fold learning-curve experiment on clear-water samples."""
    if ratios is None:
        ratios = TRAIN_RATIOS
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    X_all, y_all, wvl, meta = load_river_csv(
        csv_path,
        min_plastic_frac=min_plastic_frac,
        water_filter=water_filter,
        band_range=band_range
    )

    n_cls = len(np.unique(y_all))
    if n_cls < 2:
        raise ValueError("< 2 classes, check filters")

    records = []
    pc_records = []

    for seed_idx, seed in enumerate(seeds):
        if not quiet:
            print(f"\n{'='*60}")
            print(f"Seed {seed_idx + 1}/{len(seeds)} (seed={seed})")
            print(f"{'='*60}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
            X_train_pool = X_all[train_idx]
            y_train_pool = y_all[train_idx]
            X_test = X_all[test_idx]
            y_test = y_all[test_idx]

            fold_seed = seed + 1000 * fold_idx
            X_test, y_test = subsample_per_class(
                X_test, y_test, max_pixels_per_class=500, seed=fold_seed
            )
            X_test = X_test.astype(np.float32, copy=False)

            X_train_pool_snv = snv(X_train_pool).astype(np.float32, copy=False)

            total = len(y_train_pool)
            if not quiet:
                print(f"\n── Seed {seed_idx + 1}/{len(seeds)} Fold {fold_idx + 1}/{n_splits}  train={total}  test={len(y_test)}  classes={n_cls} ──")

            for ratio in ratios:
                n_tr = max(int(ratio * total), n_cls)
                if n_tr >= total:
                    Xtr = X_train_pool
                    Xtr_snv = X_train_pool_snv
                    ytr = y_train_pool
                else:
                    try:
                        Xtr, _, ytr, _ = train_test_split(
                            X_train_pool, y_train_pool,
                            train_size=n_tr, stratify=y_train_pool, random_state=fold_seed
                        )
                        Xtr_snv, _, _, _ = train_test_split(
                            X_train_pool_snv, y_train_pool,
                            train_size=n_tr, stratify=y_train_pool, random_state=fold_seed
                        )
                    except ValueError:
                        Xtr, _, ytr, _ = train_test_split(
                            X_train_pool, y_train_pool,
                            train_size=n_tr, random_state=fold_seed
                        )
                        Xtr_snv, _, _, _ = train_test_split(
                            X_train_pool_snv, y_train_pool,
                            train_size=n_tr, random_state=fold_seed
                        )
                Xtr = Xtr.astype(np.float32, copy=False)
                Xtr_snv = Xtr_snv.astype(np.float32, copy=False)

                for method_name, method_fn in METHODS.items():
                    try:
                        if method_name == "F: PLS-DA":
                            pls, classes = fit_plsda(Xtr, ytr)
                            y_pred = predict_plsda(pls, classes, X_test)
                            metrics = compute_metrics(y_test, y_pred)
                            pc1, pc2 = None, None
                        else:
                            is_snv_pre = method_name in SNV_PRE_METHODS
                            xtr_arg = Xtr_snv if is_snv_pre else Xtr
                            xte_arg = snv(X_test).astype(np.float32, copy=False) if is_snv_pre else X_test
                            result = method_fn(xtr_arg, xte_arg, ytr=ytr,
                                               print_variance=True, return_variance=True,
                                               pre_snv=is_snv_pre)
                            if len(result) == 4:
                                Xtr_f, Xte_f, pc1, pc2 = result
                            else:
                                Xtr_f, Xte_f = result
                                pc1, pc2 = None, None
                            skip_snv = method_name in METHODS_SKIP_POST_SNV
                            metrics = evaluate(Xtr_f, Xte_f, ytr, y_test, skip_snv=skip_snv)
                    except Exception as e:
                        metrics = {m: np.nan for m in METRIC_COLS}
                        pc1, pc2 = None, None
                        if not quiet:
                            print(f"  ⚠ {method_name} failed: {e}")

                    records.append({
                        "seed": seed,
                        "fold": fold_idx,
                        "ratio": ratio,
                        "n_train": len(ytr),
                        "method": method_name,
                        **metrics
                    })

                    if pc1 is not None:
                        pc_records.append({
                            "seed": seed,
                            "fold": fold_idx,
                            "ratio": ratio,
                            "method": method_name,
                            "PC1": pc1,
                            "PC2": pc2
                        })

                if not quiet:
                    row_a = next(r for r in records[-len(METHODS):]
                                 if r["method"] == "A: Global TruncSVD")
                    row_a2 = next(r for r in records[-len(METHODS):]
                                 if r["method"] == "A2: Global TruncSVD (SNV-Pre)")
                    print(f"  ratio={ratio:.1%} n_train={len(ytr):5d}  "
                          f"GlobalPCA OA={row_a['OA']:.3f} F1={row_a['macro_F1']:.3f}  "
                          f"SNV-Pre OA={row_a2['OA']:.3f} F1={row_a2['macro_F1']:.3f}")

    return pd.DataFrame(records), pd.DataFrame(pc_records)


def run_water_ood_experiment(csv_path,
                             ratios,
                             seeds=None,
                             n_splits=5,
                             min_plastic_frac=0.0,
                             band_range=(940, 1680),
                             train_water=0,
                             id_test_size=0.2,
                             test_max_per_class=500,
                             use_multi_train=False,
                             quiet=False):
    """Water-condition OOD experiment: train on clear, test on turbid/foamy."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    X_all, y_all, wvl, meta = load_river_csv(
        csv_path,
        min_plastic_frac=min_plastic_frac,
        water_filter=None,
        band_range=band_range
    )

    water = meta["water"].values.astype(int)
    mask_train = water == 0
    X_clear = X_all[mask_train]
    y_clear = y_all[mask_train]

    n_cls = len(np.unique(y_clear))
    if n_cls < 2:
        raise ValueError("< 2 classes in clear-water split, check filters")

    records = []

    for seed_idx, seed in enumerate(seeds):
        if not quiet:
            print(f"\n{'='*60}")
            print(f"OOD Seed {seed_idx + 1}/{len(seeds)} (seed={seed})")
            print(f"{'='*60}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        X_turb_all = X_all[water == 1]
        y_turb_all = y_all[water == 1]
        X_foam_all = X_all[water == 2]
        y_foam_all = y_all[water == 2]

        if use_multi_train:
            has_turb = len(y_turb_all) >= n_splits
            has_foam = len(y_foam_all) >= n_splits
            skf_turb = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 7)
            skf_foam = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 13)
            turb_folds = list(skf_turb.split(X_turb_all, y_turb_all)) if has_turb else None
            foam_folds = list(skf_foam.split(X_foam_all, y_foam_all)) if has_foam else None
        else:
            X_turb = X_turb_all
            y_turb = y_turb_all
            X_foam = X_foam_all
            y_foam = y_foam_all
            X_turb, y_turb = subsample_per_class(X_turb, y_turb, max_pixels_per_class=test_max_per_class, seed=seed + 2)
            X_foam, y_foam = subsample_per_class(X_foam, y_foam, max_pixels_per_class=test_max_per_class, seed=seed + 3)
            X_turb = X_turb.astype(np.float32, copy=False)
            X_foam = X_foam.astype(np.float32, copy=False)
            X_turb_illum = ood_illum(X_turb, std=0.35, seed=seed + 11).astype(np.float32, copy=False)
            X_turb_drift = ood_sensor_drift(X_turb, sigma=0.02, seed=seed + 12).astype(np.float32, copy=False)
            X_turb_shift = ood_band_shift(X_turb, shift_bands=2).astype(np.float32, copy=False)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_clear, y_clear)):
            fold_seed = seed + 1000 * fold_idx

            X_clear_train = X_clear[train_idx]
            y_clear_train = y_clear[train_idx]
            X_id = X_clear[test_idx]
            y_id = y_clear[test_idx]

            if use_multi_train:
                if turb_folds is not None:
                    tr_idx, te_idx = turb_folds[fold_idx]
                    X_turb_train = X_turb_all[tr_idx];   y_turb_train = y_turb_all[tr_idx]
                    X_turb = X_turb_all[te_idx];          y_turb = y_turb_all[te_idx]
                else:
                    X_turb_train = X_turb_all;            y_turb_train = y_turb_all
                    X_turb = X_turb_all;                  y_turb = y_turb_all
                if foam_folds is not None:
                    tr_idx, te_idx = foam_folds[fold_idx]
                    X_foam_train = X_foam_all[tr_idx];   y_foam_train = y_foam_all[tr_idx]
                    X_foam = X_foam_all[te_idx];          y_foam = y_foam_all[te_idx]
                else:
                    X_foam_train = X_foam_all;            y_foam_train = y_foam_all
                    X_foam = X_foam_all;                  y_foam = y_foam_all
                X_turb, y_turb = subsample_per_class(X_turb, y_turb, max_pixels_per_class=test_max_per_class, seed=fold_seed + 2)
                X_foam, y_foam = subsample_per_class(X_foam, y_foam, max_pixels_per_class=test_max_per_class, seed=fold_seed + 3)
                X_turb = X_turb.astype(np.float32, copy=False)
                X_foam = X_foam.astype(np.float32, copy=False)
                X_turb_illum = ood_illum(X_turb, std=0.35, seed=fold_seed + 11).astype(np.float32, copy=False)
                X_turb_drift = ood_sensor_drift(X_turb, sigma=0.02, seed=fold_seed + 12).astype(np.float32, copy=False)
                X_turb_shift = ood_band_shift(X_turb, shift_bands=2).astype(np.float32, copy=False)

            if not quiet:
                print(f"\n── Seed {seed_idx + 1}/{len(seeds)} Fold {fold_idx + 1}/{n_splits}  train={len(y_clear_train)}  test={len(y_id)} ──")

            X_id, y_id = subsample_per_class(X_id, y_id, max_pixels_per_class=test_max_per_class, seed=seed + 100 * fold_idx)
            X_id = X_id.astype(np.float32, copy=False)

            X_id_illum = ood_illum(X_id, std=0.35, seed=seed + 21 + fold_idx).astype(np.float32, copy=False)
            X_id_drift = ood_sensor_drift(X_id, sigma=0.02, seed=seed + 22 + fold_idx).astype(np.float32, copy=False)
            X_id_shift = ood_band_shift(X_id, shift_bands=2).astype(np.float32, copy=False)

            if use_multi_train:
                X_train_pool = np.vstack([X_clear_train, X_turb_train, X_foam_train])
                y_train_pool = np.concatenate([y_clear_train, y_turb_train, y_foam_train])
            else:
                X_train_pool = X_clear_train
                y_train_pool = y_clear_train

            total = len(y_train_pool)
            X_train_pool_snv = snv(X_train_pool).astype(np.float32, copy=False)

            for ratio in ratios:
                n_tr = max(int(ratio * total), n_cls)
                if n_tr >= total:
                    Xtr = X_train_pool
                    Xtr_snv = X_train_pool_snv
                    ytr = y_train_pool
                else:
                    try:
                        Xtr, _, ytr, _ = train_test_split(
                            X_train_pool, y_train_pool,
                            train_size=n_tr,
                            stratify=y_train_pool,
                            random_state=fold_seed
                        )
                        Xtr_snv, _, _, _ = train_test_split(
                            X_train_pool_snv, y_train_pool,
                            train_size=n_tr,
                            stratify=y_train_pool,
                            random_state=fold_seed
                        )
                    except ValueError:
                        Xtr, _, ytr, _ = train_test_split(
                            X_train_pool, y_train_pool,
                            train_size=n_tr,
                            random_state=fold_seed
                        )
                        Xtr_snv, _, _, _ = train_test_split(
                            X_train_pool_snv, y_train_pool,
                            train_size=n_tr,
                            random_state=fold_seed
                        )
                Xtr = Xtr.astype(np.float32, copy=False)
                Xtr_snv = Xtr_snv.astype(np.float32, copy=False)

                for method_name, method_fn in METHODS.items():
                    row = {"seed": seed, "fold": fold_idx, "ratio": ratio, "method": method_name}
                    try:
                        test_entries = [
                            (X_id, y_id, "ID"),
                            (X_turb, y_turb, "OOD-Turbid"),
                            (X_foam, y_foam, "OOD-Foamy"),
                            (X_id_illum, y_id, "ID-IllumShift"),
                            (X_id_drift, y_id, "ID-SensorDrift"),
                            (X_id_shift, y_id, "ID-BandShift"),
                            (X_turb_illum, y_turb, "OOD-IllumShift"),
                            (X_turb_drift, y_turb, "OOD-SensorDrift"),
                            (X_turb_shift, y_turb, "OOD-BandShift"),
                        ]
                        valid = [(X, y, n) for X, y, n in test_entries if len(y) > 0]

                        if method_name == "F: PLS-DA":
                            pls, classes = fit_plsda(Xtr, ytr)
                            if valid:
                                all_Xte = np.vstack([X for X, y, _ in valid])
                                all_y_pred = predict_plsda(pls, classes, all_Xte)
                                split_sizes = [len(y) for _, y, _ in valid]
                                pred_splits = np.split(all_y_pred, np.cumsum(split_sizes)[:-1])
                                for y_pred, y_cond, name in zip(pred_splits, [y for _, y, _ in valid], [n for _, _, n in valid]):
                                    row[name] = float(accuracy_score(y_cond, y_pred))
                                    row[name + "_F1"] = float(f1_score(y_cond, y_pred, average='macro'))
                            for X, y, name in test_entries:
                                if len(y) == 0:
                                    row[name] = np.nan
                                    row[name + "_F1"] = np.nan
                        else:
                            is_snv_pre = method_name in SNV_PRE_METHODS
                            xtr_arg = Xtr_snv if is_snv_pre else Xtr
                            if valid:
                                all_Xte = np.vstack([X for X, y, _ in valid])
                                if is_snv_pre:
                                    all_Xte = snv(all_Xte).astype(np.float32, copy=False)
                                split_sizes = [len(y) for _, y, _ in valid]
                                Xtr_f, all_Xte_f = method_fn(xtr_arg, all_Xte, ytr=ytr, pre_snv=is_snv_pre)
                                skip_snv = method_name in METHODS_SKIP_POST_SNV
                                if not skip_snv:
                                    Xtr_f = snv(Xtr_f)
                                    all_Xte_f = snv(all_Xte_f)
                                sc = StandardScaler()
                                clf = make_clf()
                                clf.fit(sc.fit_transform(Xtr_f), ytr)
                                te_splits = np.split(all_Xte_f, np.cumsum(split_sizes)[:-1])
                                for Xte_f, y_cond, name in zip(te_splits, [y for _, y, _ in valid], [n for _, _, n in valid]):
                                    y_pred = clf.predict(sc.transform(Xte_f))
                                    row[name] = float(accuracy_score(y_cond, y_pred))
                                    row[name + "_F1"] = float(f1_score(y_cond, y_pred, average='macro', zero_division=0))
                            for X, y, name in test_entries:
                                if len(y) == 0:
                                    row[name] = np.nan
                                    row[name + "_F1"] = np.nan
                    except Exception:
                        row.update({
                            "ID": np.nan, "OOD-Turbid": np.nan, "OOD-Foamy": np.nan,
                            "ID-IllumShift": np.nan, "ID-SensorDrift": np.nan, "ID-BandShift": np.nan,
                            "OOD-IllumShift": np.nan, "OOD-SensorDrift": np.nan, "OOD-BandShift": np.nan,
                            "ID_F1": np.nan, "OOD-Turbid_F1": np.nan, "OOD-Foamy_F1": np.nan,
                            "ID-IllumShift_F1": np.nan, "ID-SensorDrift_F1": np.nan, "ID-BandShift_F1": np.nan,
                            "OOD-IllumShift_F1": np.nan, "OOD-SensorDrift_F1": np.nan, "OOD-BandShift_F1": np.nan,
                        })

                    records.append(row)

    cols = ["seed", "fold", "ratio", "method",
            "ID", "OOD-Turbid", "OOD-Foamy",
            "ID-IllumShift", "ID-SensorDrift", "ID-BandShift",
            "OOD-IllumShift", "OOD-SensorDrift", "OOD-BandShift",
            "ID_F1", "OOD-Turbid_F1", "OOD-Foamy_F1",
            "ID-IllumShift_F1", "ID-SensorDrift_F1", "ID-BandShift_F1",
            "OOD-IllumShift_F1", "OOD-SensorDrift_F1", "OOD-BandShift_F1"]
    return pd.DataFrame(records)[cols]


def plot_ood_heatmap(df, ratio, out_dir="."):
    """Plot OOD heatmap for a given training ratio."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    conds = ["ID", "OOD-Turbid", "OOD-Foamy",
             "ID-IllumShift", "ID-SensorDrift", "ID-BandShift",
             "OOD-IllumShift", "OOD-SensorDrift", "OOD-BandShift"]
    cond_mean_cols = [f'{c}_mean' for c in conds]
    sub = df[df["ratio"] == ratio].copy()
    if sub.empty:
        return
    sub = sub.set_index("method")[cond_mean_cols]
    methods = list(sub.index)
    M = (sub.values.astype(float) * 100.0)

    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.35 * len(methods))))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"OOD (water) — OA (%)  ratio={ratio:.0%}", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    out = out_dir / f"ood_water_heatmap_ratio_{int(round(ratio*100)):02d}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def _comp_pca_fit(Xtr, n_comp, center=False, print_variance=False, method_name=""):
    """Fit TruncatedSVD on training data.

    Parameters
    ----------
    Xtr : array (n_samples, n_features)
    n_comp : int — number of components
    center : bool — if True, subtract mean before fitting (centered control)
    print_variance : bool — if True, print PC1/PC2 variance ratio
    method_name : str — method name for printing

    Returns
    -------
    svd_obj : fitted TruncatedSVD object
    mu : mean vector or None (only when center=True)
    pc1_ratio : float or None — PC1 variance ratio (0-100)
    pc2_ratio : float or None — PC2 variance ratio (0-100)
    """
    n, p = Xtr.shape
    n_comp = max(1, min(n_comp, min(n, p) - 1 if min(n, p) > 1 else 1, p))
    mu = None
    X_fit = Xtr
    if center:
        mu = Xtr.mean(axis=0)
        X_fit = Xtr - mu
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(X_fit)
    sv_sq = svd.singular_values_ ** 2
    total_energy = np.sum(X_fit ** 2)

    pc1_ratio = sv_sq[0] / total_energy * 100
    pc2_ratio = (sv_sq[1] / total_energy * 100) if len(sv_sq) > 1 else None

    return svd, mu, pc1_ratio, pc2_ratio


def _comp_pca_transform(Xte, svd_obj, mu=None):
    """Transform data using fitted TruncatedSVD. Subtract mu if provided."""
    X = Xte - mu if mu is not None else Xte
    return svd_obj.transform(X)


def method_global_pca(Xtr, Xte, n_dim=N_DIM, **kw):
    """A: Global TruncatedSVD, no centering."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr, n_dim, center=False, print_variance=print_var, method_name="A: Global PCA")
    Xtr_f, Xte_f = _comp_pca_transform(Xtr, svd, mu), _comp_pca_transform(Xte, svd, mu)
    return (Xtr_f, Xte_f, pc1, pc2) if return_var else (Xtr_f, Xte_f)

def method_global_pca_snv_pre(Xtr, Xte, n_dim=N_DIM, **kw):
    """A2: SNV → Global TruncatedSVD, no centering."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    if not kw.get('pre_snv', False):
        Xtr = snv(Xtr)
        Xte = snv(Xte)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr, n_dim, center=False, print_variance=print_var, method_name="A2: SNV-Pre")
    Xtr_f, Xte_f = _comp_pca_transform(Xtr, svd, mu), _comp_pca_transform(Xte, svd, mu)
    return (Xtr_f, Xte_f, pc1, pc2) if return_var else (Xtr_f, Xte_f)

def method_global_pca_centered(Xtr, Xte, n_dim=N_DIM, **kw):
    """A*: Global TruncatedSVD, with centering (control for A)."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr, n_dim, center=True, print_variance=print_var, method_name="A*: Centered")
    Xtr_f, Xte_f = _comp_pca_transform(Xtr, svd, mu), _comp_pca_transform(Xte, svd, mu)
    return (Xtr_f, Xte_f, pc1, pc2) if return_var else (Xtr_f, Xte_f)

def method_global_pca_snv_pre_centered(Xtr, Xte, n_dim=N_DIM, **kw):
    """A2*: SNV → Global TruncatedSVD, with centering (control for A2)."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    if not kw.get('pre_snv', False):
        Xtr = snv(Xtr)
        Xte = snv(Xte)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr, n_dim, center=True, print_variance=print_var, method_name="A2*: SNV-Pre Centered")
    Xtr_f, Xte_f = _comp_pca_transform(Xtr, svd, mu), _comp_pca_transform(Xte, svd, mu)
    return (Xtr_f, Xte_f, pc1, pc2) if return_var else (Xtr_f, Xte_f)

def method_global_pca_msc_pre(Xtr, Xte, n_dim=N_DIM, **kw):
    """A3: MSC → Global TruncatedSVD, no centering."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    msc = MSC()
    Xtr_msc = msc.fit_transform(Xtr)
    Xte_msc = msc.transform(Xte)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr_msc, n_dim, center=False, print_variance=print_var, method_name="A3: MSC-Pre")
    Xtr_f, Xte_f = _comp_pca_transform(Xtr_msc, svd, mu), _comp_pca_transform(Xte_msc, svd, mu)
    return (Xtr_f, Xte_f, pc1, pc2) if return_var else (Xtr_f, Xte_f)

def method_global_pca_msc_pre_centered(Xtr, Xte, n_dim=N_DIM, **kw):
    """A3*: MSC → Global TruncatedSVD, with centering."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    msc = MSC()
    Xtr_msc = msc.fit_transform(Xtr)
    Xte_msc = msc.transform(Xte)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr_msc, n_dim, center=True, print_variance=print_var, method_name="A3*: MSC-Pre Centered")
    Xtr_f, Xte_f = _comp_pca_transform(Xtr_msc, svd, mu), _comp_pca_transform(Xte_msc, svd, mu)
    return (Xtr_f, Xte_f, pc1, pc2) if return_var else (Xtr_f, Xte_f)

def method_global_pca_l2_row(Xtr, Xte, n_dim=N_DIM, **kw):
    """A_L2: L2 row norm (no SNV). Fit SVD then L2-normalize per-sample in feature space."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr, n_dim, center=False, print_variance=print_var, method_name="A_L2: L2 row norm")
    Ztr = _comp_pca_transform(Xtr, svd, mu)
    Zte = _comp_pca_transform(Xte, svd, mu)
    return (normalize(Ztr, norm='l2', axis=1), normalize(Zte, norm='l2', axis=1), pc1, pc2) if return_var else (normalize(Ztr, norm='l2', axis=1), normalize(Zte, norm='l2', axis=1))


def method_global_truncsvd_drop1(Xtr, Xte, n_dim=N_DIM, **kw):
    """A_drop1: Drop PC1 (use PC2–PCn) — no SNV."""
    print_var = kw.get('print_variance', False)
    return_var = kw.get('return_variance', False)
    svd, mu, pc1, pc2 = _comp_pca_fit(Xtr, n_dim, center=False, print_variance=print_var, method_name="A_drop1: Drop PC1")
    Ztr = _comp_pca_transform(Xtr, svd, mu)
    Zte = _comp_pca_transform(Xte, svd, mu)
    return (Ztr[:, 1:], Zte[:, 1:], pc1, pc2) if return_var else (Ztr[:, 1:], Zte[:, 1:])


def method_plsda(Xtr, Xte, n_components=N_DIM, **kw):
    """F: PLS-DA baseline."""
    y = kw.get("ytr", None)
    if y is None:
        raise ValueError("PLS-DA requires ytr passed as keyword argument.")
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    y_map = {c: i for i, c in enumerate(classes)}
    y_idx = np.vectorize(y_map.get)(y)
    Y = np.eye(len(classes), dtype=np.float32)[y_idx]
    k = int(min(n_components, max(1, len(classes) - 1), Xtr.shape[1], Xtr.shape[0] - 1))
    pls = PLSRegression(n_components=k)
    pls.fit(Xtr, Y)
    Xtr_f = pls.x_scores_.astype(np.float32, copy=False)
    Xte_f = pls.transform(Xte).astype(np.float32, copy=False)
    return Xtr_f, Xte_f


def fit_plsda(Xtr, ytr, n_components=N_DIM):
    """Fit PLS-DA classifier (used for direct prediction, not as DR)."""
    y = np.asarray(ytr).astype(int)
    classes = np.unique(y)
    y_map = {c: i for i, c in enumerate(classes)}
    y_idx = np.vectorize(y_map.get)(y)
    Y = np.eye(len(classes), dtype=np.float32)[y_idx]

    k = int(min(n_components, max(1, len(classes) - 1), Xtr.shape[1], Xtr.shape[0] - 1))
    if k < 1:
        raise ValueError("PLS-DA requires at least 2 training samples and >=1 component.")

    pls = PLSRegression(n_components=k)
    pls.fit(snv(Xtr), Y)
    return pls, classes


def predict_plsda(pls, classes, Xte):
    """Predict using fitted PLS-DA model."""
    Yhat = pls.predict(snv(Xte))
    idx = np.argmax(Yhat, axis=1)
    return classes[idx]


METHODS = {
    "A: Global TruncSVD": method_global_pca,
    "A*: Global TruncSVD (Centered)": method_global_pca_centered,
    "A2: Global TruncSVD (SNV-Pre)": method_global_pca_snv_pre,
    "A2*: Global TruncSVD (SNV-Pre, Centered)": method_global_pca_snv_pre_centered,
    "A3: Global TruncSVD (MSC-Pre)": method_global_pca_msc_pre,
    "A3*: Global TruncSVD (MSC-Pre, Centered)": method_global_pca_msc_pre_centered,
    "A_drop1: Drop PC1 (PC2–PCn)": method_global_truncsvd_drop1,
    "A_L2: L2 row norm (no SNV)": method_global_pca_l2_row,
    "F: PLS-DA": method_plsda,
}

METHODS_SKIP_POST_SNV = {
    "A2: Global TruncSVD (SNV-Pre)",
    "A2*: Global TruncSVD (SNV-Pre, Centered)",
    "A3: Global TruncSVD (MSC-Pre)",
    "A3*: Global TruncSVD (MSC-Pre, Centered)",
    "A_drop1: Drop PC1 (PC2–PCn)",
    "A_L2: L2 row norm (no SNV)",
    "F: PLS-DA",
}

SNV_PRE_METHODS = {
    "A2: Global TruncSVD (SNV-Pre)",
    "A2*: Global TruncSVD (SNV-Pre, Centered)",
}


def make_clf(clf_type='svm'):
    """Create a classifier.

    Parameters
    ----------
    clf_type : str
        'svm' — linear SVM (default)
        'softmax' — logistic regression (softmax)

    Returns
    -------
    clf : sklearn classifier
    """
    if clf_type == 'softmax':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, class_weight='balanced',
                                  random_state=42)
    else:
        from sklearn.svm import SVC
        return SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)


def subsample_per_class(X, y, max_pixels_per_class=None, seed=42):
    """Randomly subsample pixels to at most max_pixels_per_class per class."""
    if max_pixels_per_class is None:
        return X, y
    rng = np.random.RandomState(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > max_pixels_per_class:
            idx = rng.choice(idx, max_pixels_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    return X[keep], y[keep]


METRIC_COLS = ["OA", "AA", "macro_F1", "kappa"]


def compute_metrics(y_true, y_pred):
    """Returns dict with four standard HSI metrics."""
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "AA": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_F1": float(f1_score(y_true, y_pred,
                                   average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def snv(X, eps=1e-8):
    """Standard Normal Variate (row-wise z-score)."""
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


class MSC:
    """Multiplicative Scatter Correction."""
    def __init__(self):
        self.ref_ = None

    def fit(self, X):
        self.ref_ = X.mean(axis=0)
        return self

    def transform(self, X):
        ref = self.ref_
        ref_mean = ref.mean()
        ref_centered = ref - ref_mean
        ref_var = np.mean(ref_centered ** 2)
        X_mean = X.mean(axis=1, keepdims=True)
        cov = np.mean((X - X_mean) * ref_centered, axis=1)
        b = cov / ref_var
        a = X_mean.flatten() - b * ref_mean
        b_safe = np.where(np.abs(b) > 1e-6, b, 1.0)
        X_corr = (X - a[:, None]) / b_safe[:, None]
        return X_corr

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def evaluate(Xtr_feat, Xte_feat, ytr, yte, skip_snv=False):
    """Train classifier and compute metrics. Apply post-SNV unless skip_snv."""
    if not skip_snv:
        Xtr_feat = snv(Xtr_feat)
        Xte_feat = snv(Xte_feat)
    sc = StandardScaler()
    clf = make_clf()
    clf.fit(sc.fit_transform(Xtr_feat), ytr)
    y_pred = clf.predict(sc.transform(Xte_feat))
    return compute_metrics(yte, y_pred)


def per_class_report(Xtr_feat, Xte_feat, ytr, yte, class_names=None):
    """Return per-class precision / recall / F1 as a DataFrame."""
    Xtr_feat = snv(Xtr_feat)
    Xte_feat = snv(Xte_feat)
    sc = StandardScaler()
    clf = make_clf()
    clf.fit(sc.fit_transform(Xtr_feat), ytr)
    y_pred = clf.predict(sc.transform(Xte_feat))
    report = classification_report(yte, y_pred,
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)
    df = pd.DataFrame(report).T
    return df


TRAIN_RATIOS = [0.005, 0.01, 0.05, 0.10]


def ood_illum(X, std=0.30, seed=1):
    """Simulate illumination shift OOD perturbation."""
    rng = np.random.RandomState(seed)
    return X * (1.0 + std * rng.randn(X.shape[0], 1))


def ood_sensor_drift(X, sigma=0.10, seed=2):
    """Simulate sensor drift OOD perturbation."""
    rng = np.random.RandomState(seed)
    return X + sigma * rng.randn(1, X.shape[1])


def ood_band_shift(X, shift_bands=2):
    """Simulate band-shift OOD perturbation."""
    return np.roll(X, shift_bands, axis=1)


def plot_loading_vectors(csv_path, band_range=(940, 1680), n_dim=20, out_dir="."):
    """Plot PC1 & PC2 loading vectors from uncentered TruncatedSVD on full clear-water data.

    Parameters
    ----------
    csv_path : str or Path — path to spectradictionary.csv
    band_range : tuple — wavelength range to use (default 940–1680 nm)
    n_dim : int — number of SVD components to fit
    out_dir : str or Path — output directory for the figure
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    X_all, y_all, wvl, meta = load_river_csv(
        csv_path, water_filter=0, band_range=band_range
    )

    svd, mu, pc1_ratio, pc2_ratio = _comp_pca_fit(
        X_all, n_dim, center=False, print_variance=True, method_name="Loading Vectors"
    )
    V = svd.components_

    print(f"PC1 energy ratio: {pc1_ratio:.2f}%")
    print(f"PC2 energy ratio: {pc2_ratio:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(wvl, V[0, :], label='PC1 Loading', color='black',
            linewidth=1.8, linestyle='-')
    ax.plot(wvl, V[1, :], label='PC2 Loading', color='#d62728',
            linewidth=1.4, linestyle='--')

    ax.axvspan(1340, 1460, alpha=0.10, color='gray', label='Water absorption gap')

    ax.set_xlabel('Wavelength (nm)', fontsize=15)
    ax.set_ylabel('Loading Value (Arbitrary Units)', fontsize=15)
    ax.set_title('Loading Vectors of the First Two Uncentered PCs', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    out_path = out_dir / "fig_loading_vectors.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    return fig, ax


def main(out_dir="output",
         csv_path="spectradictionary.csv"):
    """Run all experiments: k-fold CV, OOD (single & multi-train), loading vectors."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ratios = [0.01, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    seeds = [42, 43, 44, 45, 46]

    # ── Experiment 1: Stratified 5-fold learning curves ──
    print("=" * 60)
    print("Experiment 1: Clear-water k-fold CV")
    print("=" * 60)
    df_csv, df_pc = run_csv_experiment_kfold(
        csv_path=csv_path,
        ratios=ratios,
        n_splits=5,
        seeds=seeds,
        water_filter=0,
        band_range=(940, 1680),
        quiet=False,
    )
    df_csv.to_csv(out_dir / "csv_results_raw.csv", index=False)
    print(f"\nRaw CSV results saved to {out_dir / 'csv_results_raw.csv'}")

    summary_csv = df_csv.groupby(['ratio', 'method', 'seed'])[METRIC_COLS].mean().reset_index()
    summary_csv = summary_csv.groupby(['ratio', 'method'])[METRIC_COLS].agg(['mean', 'std'])
    summary_csv.columns = [f'{col}_{stat}' for col, stat in summary_csv.columns]
    summary_csv = summary_csv.reset_index()
    summary_csv.to_csv(out_dir / "csv_results_summary.csv", index=False)
    print(f"CSV summary saved to {out_dir / 'csv_results_summary.csv'}")

    if not df_pc.empty:
        pc_summary = df_pc.groupby(['ratio', 'method'])[['PC1', 'PC2']].mean().reset_index()
        pc_summary.to_csv(out_dir / "pc_variance_summary.csv", index=False)
        print(f"PC variance summary saved to {out_dir / 'pc_variance_summary.csv'}")

    # ── Experiment 2a: OOD (single-background training) ──
    print("\n" + "=" * 60)
    print("Experiment 2a: OOD water conditions (clear-only training)")
    print("=" * 60)
    ood_ratios = [0.01, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    df_ood = run_water_ood_experiment(
        csv_path=csv_path,
        ratios=ood_ratios,
        seeds=seeds,
        min_plastic_frac=0.0,
        band_range=(940, 1680),
        train_water=0,
        id_test_size=0.2,
        test_max_per_class=500,
        use_multi_train=False,
        quiet=True
    )
    ood_cols = ["ID", "OOD-Turbid", "OOD-Foamy",
                "ID-IllumShift", "ID-SensorDrift", "ID-BandShift",
                "OOD-IllumShift", "OOD-SensorDrift", "OOD-BandShift"]
    ood_summary = df_ood.groupby(['ratio', 'method'])[ood_cols].agg(['mean', 'std'])
    ood_summary.columns = [f'{col}_{stat}' for col, stat in ood_summary.columns]
    ood_summary = ood_summary.reset_index()
    ood_csv = out_dir / "ood_water_results_summary.csv"
    ood_summary.to_csv(ood_csv, index=False)
    print(f"Saved: {ood_csv}")

    print("\n── OOD OA (mean) at ratio=100% ──")
    oa_mean_cols = [c for c in ood_summary.columns if c.endswith('_mean')]
    oa_sub = ood_summary[ood_summary["ratio"] == 1.0]
    if not oa_sub.empty:
        tbl = oa_sub.set_index("method")[oa_mean_cols].copy()
        tbl.columns = [c.replace("_mean", "") for c in tbl.columns]
        print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Experiment 2b: OOD (multi-background training) ──
    print("\n" + "=" * 60)
    print("Experiment 2b: OOD water conditions (multi-background training)")
    print("=" * 60)
    df_ood_multi = run_water_ood_experiment(
        csv_path=csv_path,
        ratios=ood_ratios,
        seeds=seeds,
        min_plastic_frac=0.0,
        band_range=(940, 1680),
        train_water=0,
        id_test_size=0.2,
        test_max_per_class=500,
        use_multi_train=True,
        quiet=True
    )
    ood_summary_multi = df_ood_multi.groupby(['ratio', 'method'])[ood_cols].agg(['mean', 'std'])
    ood_summary_multi.columns = [f'{col}_{stat}' for col, stat in ood_summary_multi.columns]
    ood_summary_multi = ood_summary_multi.reset_index()
    ood_csv_multi = out_dir / "ood_water_results_summary_multi_train.csv"
    ood_summary_multi.to_csv(ood_csv_multi, index=False)
    print(f"Saved: {ood_csv_multi}")

    print("\n── OOD Multi-Train OA (mean) at ratio=100% ──")
    oa_sub_multi = ood_summary_multi[ood_summary_multi["ratio"] == 1.0]
    if not oa_sub_multi.empty:
        tbl_multi = oa_sub_multi.set_index("method")[oa_mean_cols].copy()
        tbl_multi.columns = [c.replace("_mean", "") for c in tbl_multi.columns]
        print(tbl_multi.to_string(float_format=lambda x: f"{x:.4f}"))

    for r in ood_ratios:
        plot_ood_heatmap(ood_summary, ratio=r, out_dir=out_dir)

    # ── Loading vectors plot ──
    print("\n" + "=" * 60)
    print("Plotting Loading Vectors")
    print("=" * 60)
    plot_loading_vectors(csv_path, band_range=(940, 1680), n_dim=20, out_dir=out_dir)

    print("\nDone. Output files in:", out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="USP Pipeline: Uncentered SVD + Post-SNV")
    parser.add_argument("--csv", type=str, default="spectradictionary.csv",
                        help="Path to spectradictionary.csv")
    parser.add_argument("--out", type=str, default="output",
                        help="Output directory")
    args, _ = parser.parse_known_args()
    main(out_dir=args.out, csv_path=args.csv)
