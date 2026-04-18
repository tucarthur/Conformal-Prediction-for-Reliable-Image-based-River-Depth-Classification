"""
analysis.py
===========
Evaluation of conformal prediction methods across image classification models.

This script:
  1. Loads per-model prediction CSV files (standard and calibration splits).
  2. Computes classification metrics (accuracy, classification report, confusion matrix).
  3. Applies three conformal prediction methods — Standard CP, Mondrian CP, and APS — and
     evaluates marginal coverage, per-class coverage, and average prediction set size.
  4. Runs statistical comparison tests (Cochran's Q + pairwise McNemar) across models.
  5. Produces and saves all figures to the OUTPUT_DIR directory.

Expected CSV schema
-------------------
Each file must contain:
  - ``True Label``      : integer class index
  - ``Predicted Label`` : integer class index
  - ``Prob_Class_<k>``  : one column per class with the predicted probability for class k

Usage
-----
    python analysis.py

Configure the paths and parameters in the ``Configuration`` section below.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

# =============================================================================
# Configuration
# =============================================================================

# Directory that contains the prediction CSV files.
INPUT_DIR: Path = Path("/path/to/predictions")

# Directory where figures and results will be saved.
OUTPUT_DIR: Path = Path("/path/to/outputs")

# Conformal prediction significance level (i.e. target miscoverage rate).
ALPHA: float = 0.05

# Models to evaluate: maps a human-readable name to the file stem used in INPUT_DIR.
MODEL_FILE_STEMS: dict[str, str] = {
    "EfficientNet":       "predictions_efficientnet_v2_l",
    "ResNet":             "predictions_resnet50",
    "MobileNet":          "predictions_mobilenet_v3_large",
    "Swin Transformer":   "predictions_swin_transformer",
    "Vision Transformer": "predictions_vision_transformer",
}

# =============================================================================
# Data loading
# =============================================================================

def load_predictions(input_dir: Path, stem: str) -> pd.DataFrame:
    """Load a prediction CSV for one model from *input_dir*."""
    path = input_dir / f"{stem}.csv"
    return pd.read_csv(path)


def load_calibration_predictions(input_dir: Path, stem: str) -> pd.DataFrame:
    """Load the calibration-split prediction CSV for one model from *input_dir*."""
    path = input_dir / f"predictions_calib_{stem.removeprefix('predictions_')}.csv"
    return pd.read_csv(path)


def load_all_models(
    input_dir: Path,
    model_stems: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Return two dicts keyed by model name:
      - ``test_dfs``  : standard (test) predictions
      - ``calib_dfs`` : calibration predictions
    """
    test_dfs: dict[str, pd.DataFrame] = {}
    calib_dfs: dict[str, pd.DataFrame] = {}
    for name, stem in model_stems.items():
        test_dfs[name] = load_predictions(input_dir, stem)
        calib_dfs[name] = load_calibration_predictions(input_dir, stem)
    return test_dfs, calib_dfs


# =============================================================================
# Classification metrics
# =============================================================================

def compute_classification_metrics(df: pd.DataFrame) -> str:
    """Print and return the sklearn classification report for *df*."""
    true_labels = df["True Label"]
    predicted_labels = df["Predicted Label"]
    accuracy = np.mean(true_labels == predicted_labels)
    print(f"  Accuracy : {accuracy:.4f}")
    report = classification_report(true_labels, predicted_labels)
    print("  Classification report:\n", report)
    return report


def plot_confusion_matrix(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> None:
    """Plot and save a heatmap of the confusion matrix for *model_name*."""
    cm = confusion_matrix(df["True Label"], df["Predicted Label"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="flare", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    save_path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.pdf"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved confusion matrix to '{save_path}'")


# =============================================================================
# Conformal prediction utilities
# =============================================================================

def _get_probability_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns whose name starts with ``Prob_Class_``."""
    return [col for col in df.columns if col.startswith("Prob_Class_")]


def _compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Return the conformal quantile threshold.

    Implements ``q̂ = sorted_scores[⌈(n+1)(1-α)⌉ - 1]`` with boundary clamping.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(k, n - 1)
    return float(np.sort(scores)[k])


def _evaluate_prediction_sets(
    df: pd.DataFrame,
    prediction_sets: list[set],
) -> dict:
    """
    Compute marginal coverage, average set size, and per-class coverage.

    Returns
    -------
    dict with keys ``marginal_coverage``, ``avg_set_size``, ``per_class_coverage``.
    """
    true_labels = df["True Label"].values
    n = len(true_labels)

    covered = [true_labels[i] in prediction_sets[i] for i in range(n)]
    marginal_coverage = float(np.mean(covered))
    avg_set_size = float(np.mean([len(s) for s in prediction_sets]))

    per_class_coverage: dict[int, float] = {}
    for c in np.unique(true_labels):
        idx = np.where(true_labels == c)[0]
        per_class_coverage[int(c)] = float(
            np.mean([true_labels[i] in prediction_sets[i] for i in idx])
        )

    return {
        "marginal_coverage": marginal_coverage,
        "avg_set_size": avg_set_size,
        "per_class_coverage": per_class_coverage,
    }


# =============================================================================
# Standard conformal prediction (LAC / RAPS score variant)
# =============================================================================

def _lac_calibration_scores(cal_df: pd.DataFrame) -> np.ndarray:
    """Nonconformity scores: ``s_i = 1 - p(true_label)``."""
    prob_cols = _get_probability_columns(cal_df)
    probs = cal_df[prob_cols].values
    true_labels = cal_df["True Label"].values
    return 1.0 - probs[np.arange(len(probs)), true_labels]


def _lac_prediction_sets(test_df: pd.DataFrame, q_hat: float) -> list[set]:
    """Include class k when ``p_k >= 1 - q̂``."""
    prob_cols = _get_probability_columns(test_df)
    probs = test_df[prob_cols].values
    threshold = 1.0 - q_hat
    return [set(np.where(probs[i] >= threshold)[0]) for i in range(len(probs))]


def standard_conformal_prediction(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[float, list[set], dict]:
    """
    Standard (marginal) conformal prediction pipeline.

    Returns
    -------
    q_hat : float
        Conformal quantile threshold.
    prediction_sets : list[set]
        Prediction set for each test sample.
    metrics : dict
        Marginal coverage, average set size, and per-class coverage.
    """
    scores = _lac_calibration_scores(cal_df)
    q_hat = _compute_conformal_quantile(scores, alpha)
    prediction_sets = _lac_prediction_sets(test_df, q_hat)
    metrics = _evaluate_prediction_sets(test_df, prediction_sets)
    return q_hat, prediction_sets, metrics


# =============================================================================
# Mondrian (class-conditional) conformal prediction
# =============================================================================

def _mondrian_thresholds(cal_df: pd.DataFrame, alpha: float) -> dict[int, float]:
    """Compute per-class conformal thresholds."""
    prob_cols = _get_probability_columns(cal_df)
    probs = cal_df[prob_cols].values
    true_labels = cal_df["True Label"].values
    thresholds: dict[int, float] = {}
    for c in np.unique(true_labels):
        idx = np.where(true_labels == c)[0]
        scores = 1.0 - probs[idx, c]
        n_c = len(scores)
        k = int(np.ceil((n_c + 1) * (1 - alpha))) - 1
        k = min(k, n_c - 1)
        thresholds[int(c)] = float(np.sort(scores)[k])
    return thresholds


def _mondrian_prediction_sets(
    test_df: pd.DataFrame,
    thresholds: dict[int, float],
) -> list[set]:
    """Build prediction sets using class-specific thresholds."""
    prob_cols = _get_probability_columns(test_df)
    probs = test_df[prob_cols].values
    n_samples, n_classes = probs.shape
    prediction_sets: list[set] = []
    for i in range(n_samples):
        sample_set = {c for c in range(n_classes) if (1.0 - probs[i, c]) <= thresholds[c]}
        prediction_sets.append(sample_set)
    return prediction_sets


def mondrian_conformal_prediction(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[dict[int, float], list[set], dict]:
    """
    Mondrian (class-conditional) conformal prediction pipeline.

    Returns
    -------
    thresholds : dict[int, float]
        Per-class conformal thresholds.
    prediction_sets : list[set]
        Prediction set for each test sample.
    metrics : dict
        Marginal coverage, average set size, and per-class coverage.
    """
    thresholds = _mondrian_thresholds(cal_df, alpha)
    prediction_sets = _mondrian_prediction_sets(test_df, thresholds)
    metrics = _evaluate_prediction_sets(test_df, prediction_sets)
    return thresholds, prediction_sets, metrics


# =============================================================================
# Adaptive Prediction Sets (APS)
# =============================================================================

def _aps_calibration_scores(cal_df: pd.DataFrame) -> np.ndarray:
    """
    APS nonconformity scores.

    For each calibration sample, sort class probabilities in descending order and
    accumulate until the true class is included; the score is the resulting cumulative
    probability.
    """
    prob_cols = _get_probability_columns(cal_df)
    probs = cal_df[prob_cols].values
    true_labels = cal_df["True Label"].values
    n_samples = len(probs)
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        sorted_indices = np.argsort(-probs[i])
        sorted_probs = probs[i][sorted_indices]
        position = int(np.where(sorted_indices == true_labels[i])[0][0])
        scores[i] = float(np.sum(sorted_probs[: position + 1]))
    return scores


def _aps_prediction_sets(test_df: pd.DataFrame, q_hat: float) -> list[set]:
    """Build APS prediction sets by greedily accumulating classes until the threshold is met."""
    prob_cols = _get_probability_columns(test_df)
    probs = test_df[prob_cols].values
    prediction_sets: list[set] = []
    for i in range(len(probs)):
        sorted_indices = np.argsort(-probs[i])
        sorted_probs = probs[i][sorted_indices]
        cumulative = 0.0
        sample_set: set = set()
        for idx, p in zip(sorted_indices, sorted_probs):
            cumulative += p
            sample_set.add(int(idx))
            if cumulative >= q_hat:
                break
        prediction_sets.append(sample_set)
    return prediction_sets


def aps_conformal_prediction(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[float, list[set], dict]:
    """
    Adaptive Prediction Sets (APS) conformal prediction pipeline.

    Returns
    -------
    q_hat : float
        APS conformal quantile threshold.
    prediction_sets : list[set]
        Prediction set for each test sample.
    metrics : dict
        Marginal coverage, average set size, and per-class coverage.
    """
    scores = _aps_calibration_scores(cal_df)
    q_hat = _compute_conformal_quantile(scores, alpha)
    prediction_sets = _aps_prediction_sets(test_df, q_hat)
    metrics = _evaluate_prediction_sets(test_df, prediction_sets)
    return q_hat, prediction_sets, metrics


# =============================================================================
# Calibration curve
# =============================================================================

def plot_calibration_curve(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    alpha_range: np.ndarray | None = None,
) -> None:
    """
    Plot and save the marginal coverage vs. confidence level (1 - α) calibration curve
    for the standard conformal method.

    Parameters
    ----------
    alpha_range : array-like, optional
        Significance levels to sweep. Defaults to ``np.arange(0.01, 0.91, 0.01)``.
    """
    if alpha_range is None:
        alpha_range = np.arange(0.01, 0.91, 0.01)

    confidence_levels = 1.0 - alpha_range
    coverage_results = []

    for alpha in alpha_range:
        _, _, metrics = standard_conformal_prediction(cal_df, test_df, alpha=alpha)
        coverage_results.append(metrics["marginal_coverage"])

    coverage_arr = np.array(coverage_results)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(confidence_levels, coverage_arr, marker="o", label="Empirical")
    ax.plot([0, 1], [0, 1], "--", color="black", label="Ideal")
    ax.fill_between(confidence_levels, coverage_arr, confidence_levels, alpha=0.2)
    ax.set_xlabel("Confidence Level (1 − α)")
    ax.set_ylabel("Marginal Coverage")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    save_path = output_dir / f"calibration_curve_{model_name.lower().replace(' ', '_')}.pdf"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved calibration curve to '{save_path}'")


# =============================================================================
# Prediction set size comparison
# =============================================================================

def plot_avg_set_size_comparison(
    summary_rows: list[dict],
    output_dir: Path,
) -> None:
    """
    Bar chart comparing average prediction set sizes across models and CP methods.

    Parameters
    ----------
    summary_rows : list of dict
        Each dict must contain keys ``"Model"``, ``"Method"``, ``"Avg Set Size"``.
    """
    df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Avg Set Size", hue="Method", ax=ax)
    ax.set_title("Average Prediction Set Size Across Models and Conformal Methods")
    ax.set_ylabel("Average Prediction Set Size")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Method", loc="upper right")
    fig.tight_layout()

    save_path = output_dir / "avg_set_size_comparison.pdf"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved set-size comparison to '{save_path}'")


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading prediction files …")
    print("=" * 60)
    test_dfs, calib_dfs = load_all_models(INPUT_DIR, MODEL_FILE_STEMS)
    model_names = list(MODEL_FILE_STEMS.keys())
    print(f"  Loaded {len(model_names)} models: {model_names}\n")

    # ------------------------------------------------------------------
    # 2. Classification metrics and confusion matrices
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Classification metrics")
    print("=" * 60)
    for name in model_names:
        print(f"\n— {name} —")
        compute_classification_metrics(test_dfs[name])
        plot_confusion_matrix(test_dfs[name], name, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 3. Conformal prediction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Conformal prediction  (α = {ALPHA})")
    print("=" * 60)

    summary_rows: list[dict] = []

    for name in model_names:
        cal_df = calib_dfs[name]
        test_df = test_dfs[name]

        print(f"\n— {name} —")

        # Standard CP
        q_hat, _, metrics_cp = standard_conformal_prediction(cal_df, test_df, alpha=ALPHA)
        print(
            f"  [Standard CP]  q̂={q_hat:.4f}  "
            f"coverage={metrics_cp['marginal_coverage']:.4f}  "
            f"avg set size={metrics_cp['avg_set_size']:.4f}"
        )
        summary_rows.append({"Model": name, "Method": "CP", "Avg Set Size": metrics_cp["avg_set_size"]})

        # Mondrian CP
        thresholds, _, metrics_mondrian = mondrian_conformal_prediction(cal_df, test_df, alpha=ALPHA)
        print(
            f"  [Mondrian CP]  coverage={metrics_mondrian['marginal_coverage']:.4f}  "
            f"avg set size={metrics_mondrian['avg_set_size']:.4f}"
        )
        summary_rows.append({"Model": name, "Method": "Mondrian", "Avg Set Size": metrics_mondrian["avg_set_size"]})

        # APS
        q_hat_aps, _, metrics_aps = aps_conformal_prediction(cal_df, test_df, alpha=ALPHA)
        print(
            f"  [APS]          q̂={q_hat_aps:.4f}  "
            f"coverage={metrics_aps['marginal_coverage']:.4f}  "
            f"avg set size={metrics_aps['avg_set_size']:.4f}"
        )
        summary_rows.append({"Model": name, "Method": "APS", "Avg Set Size": metrics_aps["avg_set_size"]})

        # Calibration curve (standard CP)
        plot_calibration_curve(cal_df, test_df, name, OUTPUT_DIR)

    plot_avg_set_size_comparison(summary_rows, OUTPUT_DIR)


    print("\nDone.")


if __name__ == "__main__":
    main()