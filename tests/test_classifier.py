"""Tests for the email classifiers (``mailmind.ml.classifier``)."""
from __future__ import annotations

import numpy as np

from mailmind import config
from mailmind.ml.classifier import HeuristicClassifier, MailMindClassifier
from mailmind.schema import Classification, Email


# --------------------------------------------------------------------------- #
# Heuristic classifier
# --------------------------------------------------------------------------- #
def test_heuristic_classify_returns_valid_classification():
    """The heuristic classifier returns a valid Classification."""
    clf = HeuristicClassifier()
    result = clf.classify(
        Email(subject="meeting agenda", body="project review with the team")
    )
    assert isinstance(result, Classification)
    assert result.label in config.CATEGORIES
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities.keys()) == set(config.CATEGORIES)


def test_heuristic_predict_batch():
    """Batch prediction returns one in-vocabulary label per email."""
    clf = HeuristicClassifier()
    emails = [
        Email(subject="win a free prize", body="claim your lottery cash now"),
        Email(subject="quarterly report", body="deadline for the client deliverable"),
    ]
    labels = clf.predict(emails)
    assert len(labels) == 2
    assert all(label in config.CATEGORIES for label in labels)


# --------------------------------------------------------------------------- #
# ML classifier
# --------------------------------------------------------------------------- #
def test_trained_classifier_is_fitted(trained_classifier):
    """A trained classifier reports itself as fitted."""
    assert trained_classifier.is_fitted is True
    assert set(trained_classifier.labels) == set(config.CATEGORIES)


def test_trained_classifier_predict_and_proba(trained_classifier, sample_emails):
    """predict returns labels and predict_proba has shape (n, n_labels)."""
    emails = list(sample_emails.values())
    labels = trained_classifier.predict(emails)
    assert len(labels) == len(emails)
    assert all(label in config.CATEGORIES for label in labels)

    proba = trained_classifier.predict_proba(emails)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(emails), len(trained_classifier.labels))
    # Each row is a valid probability distribution.
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_trained_classifier_classify_confidence(trained_classifier, sample_email):
    """classify() returns a Classification with confidence in [0, 1]."""
    result = trained_classifier.classify(sample_email)
    assert isinstance(result, Classification)
    assert result.label in config.CATEGORIES
    assert 0.0 <= result.confidence <= 1.0


def test_trained_classifier_save_load_roundtrip(trained_classifier, tmp_path, sample_email):
    """A saved classifier reloads and produces identical predictions."""
    path = tmp_path / "model.joblib"
    trained_classifier.save(path)
    assert path.exists()

    reloaded = MailMindClassifier.load(path)
    assert reloaded.is_fitted is True
    assert reloaded.labels == trained_classifier.labels
    assert reloaded.classify(sample_email).label == trained_classifier.classify(
        sample_email
    ).label
