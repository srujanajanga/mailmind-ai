"""Machine-learning subsystem for MailMind AI.

Bundles feature engineering, the email classifiers, evaluation utilities, and
the training driver behind a single import surface::

    from mailmind.ml import MailMindClassifier, train

Public API:
    * :class:`MailMindClassifier` / :class:`HeuristicClassifier`
    * :func:`compute_engineered_features`, :func:`emails_to_frame`,
      :func:`build_preprocessor`
    * :func:`evaluate_predictions`
    * :func:`train`
"""
from __future__ import annotations

from mailmind.ml.classifier import HeuristicClassifier, MailMindClassifier
from mailmind.ml.evaluate import evaluate_predictions
from mailmind.ml.features import (
    build_preprocessor,
    compute_engineered_features,
    emails_to_frame,
)
from mailmind.ml.train import train

__all__ = [
    "MailMindClassifier",
    "HeuristicClassifier",
    "emails_to_frame",
    "compute_engineered_features",
    "build_preprocessor",
    "evaluate_predictions",
    "train",
]
