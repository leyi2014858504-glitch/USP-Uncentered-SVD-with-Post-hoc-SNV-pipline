# USP Pipeline: Uncentered SVD + Post-SNV for SWIR Plastic Classification

Research pipeline for the paper *"Uncentered SVD + Post-SNV for SWIR Plastic Classification"*.

## Project Structure

```
PCA/
├── usp_pipeline.py              # Main pipeline script
├── usp-pipline (1).ipynb        # Colab-compatible notebook
├── run_textile_experiment.py    # Cross-domain textile validation
├── plot_results.py              # Figure generation
├── 1/
│   ├── README.md                # Data download instructions
│   ├── spectradictionary.csv    # Plastic SWIR dataset (193 samples) — download from Zenodo
│   └── swir_mean_spectra.csv    # Textile SWIR dataset — download from Zenodo
├── ground_truth_final.csv       # Textile fiber composition labels — download from Zenodo
├── csv_results_summary.csv      # Clear-water k-fold CV results
├── ood_water_results_summary.csv          # OOD single-train results
├── ood_water_results_summary_multi_train.csv  # OOD multi-train results
├── pc_variance_summary.csv      # PC energy ratios
├── textile_results_summary.csv  # Textile LOO-CV results
└── fig_*.png                    # All figures
```

## Methods

| Method | Description |
|--------|-------------|
| A | Global TruncSVD (no centering) — **proposed** |
| A* | Global TruncSVD (centered) — ablation |
| A2 | SNV-Pre + TruncSVD (no centering) |
| A2* | SNV-Pre + TruncSVD (centered) |
| A3 | MSC-Pre + TruncSVD (no centering) |
| A3* | MSC-Pre + TruncSVD (centered) |
| A_drop1 | TruncSVD with PC1 dropped |
| A_L2 | TruncSVD with L2 normalization |
| F | PLS-DA (baseline) |

## Scripts

### 1. `usp_pipeline.py` — Main Pipeline

Runs all core experiments:

- **Clear-water k-fold CV**: Stratified 5-fold learning curves over training ratios [0.01, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00] across 5 seeds. Outputs `csv_results_summary.csv` and `pc_variance_summary.csv`.

- **OOD water conditions (single-train)**: Train on Clear water only, test on Turbid/Foamy + simulated perturbations (illumination shift, sensor drift, band shift). Outputs `ood_water_results_summary.csv`.

- **OOD water conditions (multi-train)**: Train on Clear + Turbid + Foamy jointly, evaluate held-out ID split and simulated perturbations. Outputs `ood_water_results_summary_multi_train.csv`.

- **Loading vectors plot**: PC1 & PC2 loading vectors from uncentered SVD on full clear-water data with water absorption gap shading (1340–1460 nm). Outputs `fig_loading_vectors.png`.

#### Usage

```bash
# Default: read 1/spectradictionary.csv, output to output/
python usp_pipeline.py

# Custom paths
python usp_pipeline.py --csv path/to/spectradictionary.csv --out results/
```

#### Switching Classifier

The `make_clf()` function supports two classifier types via the `clf_type` parameter:

| `clf_type` | Classifier | Description |
|-----------|-----------|-------------|
| `'svm'` (default) | `SVC(kernel='linear', C=1.0)` | Linear SVM with balanced class weight |
| `'softmax'` | `LogisticRegression(max_iter=2000)` | Logistic regression (softmax) |

To change the classifier globally, modify the callers in `evaluate()` and `per_class_report()`:

```python
clf = make_clf(clf_type='softmax')  # switch to logistic regression
```

---

### 2. `usp-pipline (1).ipynb` — Kaggle Notebook

Jupyter version of `usp_pipeline.py`. All functions and experiments are identical. Compatible with Kaggle notebook (`parse_known_args()` handles kernel-injected arguments).

> **Note:** This notebook can reproduce the paper results numerically (1:1), but it differs from the publicly shared Kaggle version in that it contains some redundant code and unreported experiments that were removed in the cleaned `usp_pipeline.py`.

#### Usage

Open in Jupyter or upload to Kaggle notebook. Run all cells. Output files are written to the `output/` directory.

---

### 3. `run_textile_experiment.py` — Cross-Domain Validation

Validates the proposed method on the OpenTextile SWIR dataset. Compares four methods via Leave-One-Out CV:

- **A** (Proposed): TruncSVD + post-SNV
- **A2**: SNV-Pre + TruncSVD (no post-SNV)
- **A_noSNV**: TruncSVD without any SNV
- **F**: PLS-DA

Data preprocessing:
- 71 samples → filter unknown/outlier → 60 usable
- Dominant fiber label (≥50% threshold)
- Band range: 1000–1680 nm

Outputs `textile_results_summary.csv` and `fig_textile_classification.png`.

#### Usage

```bash
python run_textile_experiment.py
```

Requires `1/swir_mean_spectra.csv` and `ground_truth_final.csv` in the workspace.

---

### 4. `plot_results.py` — Figure Generation

Generates all paper figures from the summary CSV files:

| Function | Input | Output |
|----------|-------|--------|
| `plot_clear_only_oa()` | `csv_results_summary.csv` | `fig_clear_only_oa.png` |
| `plot_ood_oa()` | `ood_water_results_summary.csv` | `fig_ood_oa_separate.png` |
| `plot_ood_combined()` | OOD summary CSV | Combined OOD figure |
| `plot_pc1_variance()` | `pc_variance_summary.csv` | PC energy ratio bar charts |
| `plot_pc1_variance_single_ratio()` | `pc_variance_summary.csv` | `fig_pc1_variance_ratio100.png` |
| `plot_loading_vectors()` | Full spectra data | Loading vectors plot |
| `plot_textile_pc1_verification()` | PC variance data | Textile PC energy verification |

#### Usage

```python
# From within a Python script or notebook
from plot_results import plot_clear_only_oa, plot_ood_oa

plot_clear_only_oa("csv_results_summary.csv", out_dir="output")
plot_ood_oa("ood_water_results_summary.csv", out_dir="output")
```

Or run specific plotting functions from the command line by adding a `__main__` block (functions are imported, not executed).

## Data Format

### `spectradictionary.csv` (Plastic Dataset)

- Rows: samples (193 plastic spectra + metadata)
- Columns: wavelength bands (350–2500 nm) + metadata columns (`water`, `Description`, `Class2`)
- `water` column: 0=Clear, 1=Turbid, 2=Foamy
- `Class2` column: material label (PE, PP, PS, PET, PLA, PA, PC, etc.)
- Pipeline uses band range 940–1680 nm

### `swir_mean_spectra.csv` (Textile Dataset)

- Rows: sample IDs
- Columns: wavelength bands (as numeric strings, e.g. `1000.0`, `1003.0`, ...)
- Separator: `;`

### `ground_truth_final.csv` (Textile Labels)

- Columns: fiber composition percentages (e.g. `Cotton [%]`, `Polyester [%]`)
- Additional metadata: `Unknown or outlier?` column for filtering

## Data Sources

The datasets used in this project are publicly available on Zenodo. Please download them from the following links and place the files in the paths expected by the scripts.

| Dataset | Files Needed | Path |
|---------|-------------|------|
| [A Hyperspectral Reflectance Database of Plastic Debris for River Ecosystems](https://zenodo.org/records/13377060) | `spectradictionary.csv` | `1/spectradictionary.csv` |
| [OpenTextile-NIR: Near-infrared hyperspectral imaging and photography dataset for optical identification of textiles](https://zenodo.org/records/18269172) | `swir_mean_spectra.csv`, `ground_truth_final.csv` | `1/swir_mean_spectra.csv`, `ground_truth_final.csv` |

> **Note:** The dataset files themselves are not included in this repository. Please cite the original sources when using them.

## Metrics

All experiments report:
- **OA** — Overall Accuracy
- **AA** — Average (Balanced) Accuracy
- **macro_F1** — Macro-averaged F1 score
- **κ (kappa)** — Cohen's Kappa

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
```

## Reproducibility

All experiments use fixed random seeds (42–46). Results in the summary CSV files were generated by running `usp_pipeline.py` and `run_textile_experiment.py` with default parameters.
