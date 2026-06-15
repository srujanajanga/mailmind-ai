"""Training driver for the MailMind email classifier.

Running this module trains and compares several scikit-learn estimators on the
synthetic email corpus, selects the best by macro-F1, persists the winning
pipeline, and writes metrics, a model-comparison table, and evaluation figures
into the locations defined in :mod:`mailmind.config`.

Usage::

    PYTHONPATH=src python3 -m mailmind.ml.train --samples 700 --seed 42
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from mailmind import config
from mailmind.ml.classifier import MailMindClassifier
from mailmind.ml.evaluate import (
    evaluate_predictions,
    plot_category_distribution,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_model_comparison,
)
from mailmind.ml.features import build_preprocessor, emails_to_frame


# --------------------------------------------------------------------------- #
# Candidate estimators
# --------------------------------------------------------------------------- #
def _candidates(seed: int) -> dict[str, object]:
    """Return the dictionary of candidate estimators to benchmark.

    ``LinearSVC`` is wrapped in :class:`CalibratedClassifierCV` so it exposes a
    ``predict_proba`` like the other estimators, keeping the downstream
    :class:`MailMindClassifier` uniform.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=4.0, random_state=seed
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=seed), cv=3
        ),
        "ComplementNB": ComplementNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed
        ),
    }


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def _load_dataframe(samples: Optional[int] = None) -> pd.DataFrame:
    """Load (or generate) the labelled email corpus as a DataFrame.

    Prefers an existing CSV at :data:`config.DATASET_PATH`; otherwise delegates
    to :mod:`mailmind.data` to synthesise one. The ``mailmind.data`` import is
    deferred so this module stays importable while sibling subsystems are still
    being written.
    """
    from mailmind import data as data_module  # local import by design

    if samples is None and config.DATASET_PATH.exists():
        df = pd.read_csv(config.DATASET_PATH)
    else:
        per_category = samples if samples is not None else config.SAMPLES_PER_CATEGORY
        df = data_module.generate_dataset(per_category)

    if "clean_text" not in df.columns:
        df = data_module.add_clean_text(df)
    return df


def _frame_from_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model design matrix from raw dataset rows."""
    return emails_to_frame(df.to_dict("records"))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    df: Optional[pd.DataFrame] = None,
    seed: int = config.RANDOM_SEED,
    samples: Optional[int] = None,
) -> dict:
    """Train, compare, and persist the best email classifier.

    Args:
        df: Pre-loaded labelled DataFrame (must contain a ``label`` column). If
            ``None``, the corpus is loaded/generated automatically.
        seed: Random seed for reproducible splits and estimators.
        samples: Samples-per-category override when generating the corpus.

    Returns:
        A summary dict with the winning model name, its metrics, the per-model
        comparison rows, and the artefact paths that were written.
    """
    if df is None:
        df = _load_dataframe(samples)

    X = _frame_from_rows(df)
    y = df["label"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    comparison_rows: list[dict] = []
    best_name: Optional[str] = None
    best_f1 = -1.0
    best_pipeline: Optional[Pipeline] = None
    best_metrics: dict = {}

    for name, estimator in _candidates(seed).items():
        pipeline = Pipeline(
            steps=[("features", build_preprocessor()), ("estimator", estimator)]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, labels=config.CATEGORIES)

        comparison_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    assert best_pipeline is not None and best_name is not None  # for type-checkers

    # ----------------------------------------------------------------- #
    # Persist the winning model + artefacts
    # ----------------------------------------------------------------- #
    classifier = MailMindClassifier(pipeline=best_pipeline)
    fitted_classes = [str(c) for c in best_pipeline.steps[-1][1].classes_]
    ordered = [c for c in config.CATEGORIES if c in fitted_classes]
    classifier.labels = ordered + [c for c in fitted_classes if c not in ordered]
    classifier.save(config.MODEL_PATH)

    metrics_payload = {"best_model": best_name, **best_metrics}
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "f1_macro", ascending=False
    )
    comparison_df.to_csv(config.COMPARISON_PATH, index=False)

    # ----------------------------------------------------------------- #
    # Figures
    # ----------------------------------------------------------------- #
    figure_paths = {
        "confusion_matrix": plot_confusion_matrix(
            best_metrics["confusion_matrix"], config.CATEGORIES
        ),
        "per_class_metrics": plot_metric_bars(best_metrics["per_class"]),
        "category_distribution": plot_category_distribution(df),
        "model_comparison": plot_model_comparison(comparison_df),
    }

    return {
        "best_model": best_name,
        "accuracy": best_metrics["accuracy"],
        "f1_macro": best_metrics["f1_macro"],
        "f1_weighted": best_metrics["f1_weighted"],
        "per_class": best_metrics["per_class"],
        "comparison": comparison_rows,
        "model_path": str(config.MODEL_PATH),
        "metrics_path": str(config.METRICS_PATH),
        "comparison_path": str(config.COMPARISON_PATH),
        "figures": {k: str(v) for k, v in figure_paths.items()},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_report(summary: dict) -> None:
    """Print a concise, human-readable training report."""
    print(f"Best model : {summary['best_model']}")
    print(f"Accuracy   : {summary['accuracy']:.4f}")
    print(f"Macro F1   : {summary['f1_macro']:.4f}")
    print(f"Weighted F1: {summary['f1_weighted']:.4f}")
    print("Per-class F1:")
    for label in config.CATEGORIES:
        cls = summary["per_class"].get(label)
        if cls is not None:
            print(f"  {label:<12} {cls['f1-score']:.4f}")
    print(f"Model saved to: {summary['model_path']}")


def main() -> None:
    """Command-line entry point: train and report."""
    parser = argparse.ArgumentParser(description="Train the MailMind classifier.")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Samples per category to generate (default: config.SAMPLES_PER_CATEGORY).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    summary = train(seed=args.seed, samples=args.samples)
    _print_report(summary)


if __name__ == "__main__":
    main()
