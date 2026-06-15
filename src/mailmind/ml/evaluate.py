"""Evaluation metrics and publication-ready figures for the classifier.

All plotting goes through a non-interactive Matplotlib backend so the functions
are safe to call from training scripts, notebooks, or a head-less CI runner.
Every plot helper saves a tight, high-DPI PNG and returns the destination
:class:`~pathlib.Path` for convenient chaining.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from mailmind import config  # noqa: E402


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """Compute the standard suite of multi-class metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Label ordering for the report / confusion matrix. Defaults to
            :data:`mailmind.config.CATEGORIES`.

    Returns:
        A dict with accuracy, macro/weighted precision/recall/F1, the full
        per-class ``classification_report`` dict, and the confusion matrix as a
        list of lists.
    """
    labels = list(labels) if labels is not None else list(config.CATEGORIES)

    accuracy = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": accuracy,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _finalise(fig: plt.Figure, path: "str | Path") -> Path:
    """Save ``fig`` tightly at high DPI and close it; return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm: "Sequence[Sequence[float]] | np.ndarray",
    labels: Sequence[str],
    path: "str | Path" = config.FIGURES_DIR / "confusion_matrix.png",
) -> Path:
    """Plot a row-normalised confusion matrix annotated with raw counts."""
    cm_arr = np.asarray(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.divide(
            cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0
        )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = 0.5
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            colour = "white" if normalised[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{int(cm_arr[i, j])}\n{normalised[i, j]:.0%}",
                ha="center",
                va="center",
                color=colour,
                fontsize=9,
            )
    return _finalise(fig, path)


def plot_metric_bars(
    report_dict: dict,
    path: "str | Path" = config.FIGURES_DIR / "per_class_metrics.png",
) -> Path:
    """Plot grouped per-class precision / recall / F1 bars."""
    classes = [
        key
        for key in report_dict
        if key not in {"accuracy", "macro avg", "weighted avg"}
    ]
    precision = [report_dict[c]["precision"] for c in classes]
    recall = [report_dict[c]["recall"] for c in classes]
    f1 = [report_dict[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * len(classes)), 5.5))
    ax.bar(x - width, precision, width, label="Precision", color="#1d3557")
    ax.bar(x, recall, width, label="Recall", color="#2a9d8f")
    ax.bar(x + width, f1, width, label="F1-score", color="#e9c46a")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _finalise(fig, path)


def plot_category_distribution(
    df: pd.DataFrame,
    path: "str | Path" = config.FIGURES_DIR / "category_distribution.png",
    label_column: str = "label",
) -> Path:
    """Plot the count of emails per category using the project colour map."""
    counts = df[label_column].value_counts()
    ordered = [c for c in config.CATEGORIES if c in counts.index]
    extras = [c for c in counts.index if c not in ordered]
    categories = ordered + list(extras)
    values = [int(counts[c]) for c in categories]
    colours = [config.CATEGORY_COLORS.get(c, "#888888") for c in categories]

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    bars = ax.bar(x, values, color=colours)
    ax.set_ylabel("Number of emails")
    ax.set_title("Category Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _finalise(fig, path)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    path: "str | Path" = config.FIGURES_DIR / "model_comparison.png",
) -> Path:
    """Plot macro-F1 and accuracy per candidate model as grouped bars."""
    df = comparison_df.copy()
    name_col = "model" if "model" in df.columns else df.columns[0]
    models = df[name_col].astype(str).tolist()

    f1_col = "f1_macro" if "f1_macro" in df.columns else None
    acc_col = "accuracy" if "accuracy" in df.columns else None

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(models)), 5.5))
    if f1_col:
        ax.bar(x - width / 2, df[f1_col], width, label="Macro F1", color="#e63946")
    if acc_col:
        ax.bar(x + width / 2, df[acc_col], width, label="Accuracy", color="#457b9d")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _finalise(fig, path)
