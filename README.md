# Conformal Prediction for Image Classification

Companion code for the paper **"[Paper Title]"**.

This repository contains scripts for training five ImageNet pre-trained backbones on a 4-class image dataset and evaluating them under three conformal prediction frameworks: Standard CP, Mondrian CP, and Adaptive Prediction Sets (APS). Statistical comparison across models is performed via Cochran's Q and pairwise McNemar tests.

---

## Repository structure

```
.
├── train.py               # Data splitting, model training, and k-fold grid search
├── analysis.py            # Conformal prediction evaluation and statistical tests
└── predictions/           # Pre-computed prediction CSVs (one per model)
    ├── predictions_efficientnet_v2_l.csv
    ├── predictions_resnet50.csv
    ├── predictions_mobilenet_v3_large.csv
    ├── predictions_swin_transformer.csv
    ├── predictions_vision_transformer.csv
    ├── predictions_calib_efficientnet_v2_l.csv
    ├── predictions_calib_resnet50.csv
    ├── predictions_calib_mobilenet_v3_large.csv
    ├── predictions_calib_swin_transformer.csv
    └── predictions_calib_vision_transformer.csv
```

---

## Requirements

```
Python >= 3.10
torch >= 2.0
torchvision >= 0.15
numpy
pandas
scikit-learn
statsmodels
matplotlib
seaborn
tqdm
Pillow
```

Install all dependencies with:

```bash
pip install torch torchvision numpy pandas scikit-learn statsmodels matplotlib seaborn tqdm Pillow
```

---

## Data

Images should be organised in two flat directories (train and test) with file names following the format `<label>_<anything>.jpg`. The label must be one of `low`, `medium`, `high`, or any other string (mapped to class 3).

The script merges both folders and re-splits them into stratified train / validation / test subsets (60 / 20 / 20).

---

## Usage

### 1 — Training (`train.py`)

Set the paths and hyper-parameters in the **Configuration** block at the top of `train.py`:

| Variable | Description |
|---|---|
| `TRAIN_FOLDER` | Path to the original training images |
| `TEST_FOLDER` | Path to the original test images |
| `OUTPUT_DIR` | Directory for checkpoints, prediction CSVs, and the summary CSV |
| `BACKBONE_CONFIGS` | Per-backbone hyper-parameters used in the final training run |
| `KFOLD_HYPERPARAMS` | Search space for the k-fold grid search |
| `KFOLD_BACKBONES` | List of backbones to include in the grid search |

Then run:

```bash
python train.py
```

This will:
1. Build the stratified train / val / test split.
2. Run a 5-fold stratified grid search for each backbone to select the best hyper-parameters.
3. Re-train each backbone on the full training set with the selected hyper-parameters.
4. Save per-sample predictions to `predictions_calib_<backbone>.csv` and a summary to `model_results_summary.csv`.

### 2 — Analysis (`analysis.py`)

Set the paths in the **Configuration** block at the top of `analysis.py`:

| Variable | Description |
|---|---|
| `INPUT_DIR` | Path to the folder containing the prediction CSVs (e.g. `predictions/`) |
| `OUTPUT_DIR` | Directory for figures and the McNemar results CSV |
| `ALPHA` | Conformal prediction significance level (default: `0.05`) |

Then run:

```bash
python analysis.py
```

This will:
1. Compute accuracy, classification reports, and confusion matrices for each model.
2. Apply Standard CP, Mondrian CP, and APS at the specified `ALPHA` level.
3. Plot calibration curves and a bar chart comparing average prediction set sizes across methods.
4. Run Cochran's Q and pairwise McNemar tests across all models.
5. Save all figures as PDF and the McNemar results as `pairwise_mcnemar_results.csv`.

---

## Prediction CSV format

Each CSV in `predictions/` must contain the following columns:

| Column | Description |
|---|---|
| `True Label` | Integer ground-truth class index |
| `Predicted Label` | Integer predicted class index |
| `Prob_Class_0` … `Prob_Class_K` | Softmax probability for each class |

Files without the `predictions_calib_` prefix are the test-set predictions used for conformal evaluation. Files with the prefix are the calibration-set predictions used to fit the conformal thresholds.

---

## Outputs

| File | Description |
|---|---|
| `<backbone>_best.pth` | Best model checkpoint (saved by validation accuracy) |
| `predictions_calib_<backbone>.csv` | Per-sample predictions on the calibration set |
| `model_results_summary.csv` | Test accuracy and loss for all backbones |
| `confusion_matrix_<backbone>.pdf` | Confusion matrix heatmap |
| `calibration_curve_<backbone>.pdf` | Marginal coverage vs. confidence level curve |
| `avg_set_size_comparison.pdf` | Bar chart of average set sizes across models and methods |
| `pairwise_mcnemar_results.csv` | Pairwise McNemar test statistics and p-values |

---

## Reproducibility

All scripts fix seeds for Python, NumPy, PyTorch (CPU and CUDA), and CuBLAS via `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`. The global seed defaults to `42` and is set in the `SEED` variable in each script's Configuration block.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{[citekey],
  title   = {[Paper Title]},
  author  = {[Authors]},
  booktitle = {[Venue]},
  year    = {[Year]}
}
```
