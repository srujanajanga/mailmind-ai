"""Tests for the feature-engineering layer (``mailmind.ml.features``)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer

from mailmind import config
from mailmind.ml.features import (
    build_preprocessor,
    compute_engineered_features,
    emails_to_frame,
)
from mailmind.schema import Email


def test_compute_engineered_features_keys():
    """The engineered feature dict has exactly the configured numeric keys."""
    email = Email(subject="Hello", body="This is a short body with a link http://x.com")
    feats = compute_engineered_features(email)
    assert set(feats.keys()) == set(config.NUMERIC_FEATURES)
    assert all(isinstance(v, float) for v in feats.values())


def test_compute_engineered_features_signal_values():
    """The cue-count features react to obvious signals in the text."""
    email = Email(
        subject="URGENT deadline",
        body="Act now!!! Free $100 offer ASAP",
        has_attachment=True,
        num_links=3,
    )
    feats = compute_engineered_features(email)
    assert feats["has_attachment"] == 1.0
    assert feats["num_links"] == 3.0
    assert feats["exclaim_count"] >= 3.0
    assert feats["urgency_hits"] >= 1.0
    assert feats["money_hits"] >= 1.0


def test_emails_to_frame_columns():
    """The design frame carries clean_text plus all numeric feature columns."""
    emails = [
        Email(subject="A", body="alpha beta"),
        {"subject": "B", "body": "gamma"},
        "raw string email body",
    ]
    frame = emails_to_frame(emails)
    assert list(frame.columns) == ["clean_text", *config.NUMERIC_FEATURES]
    assert len(frame) == 3


def test_emails_to_frame_empty():
    """An empty input still yields the canonical (empty) frame schema."""
    frame = emails_to_frame([])
    assert list(frame.columns) == ["clean_text", *config.NUMERIC_FEATURES]
    assert len(frame) == 0


def test_build_preprocessor_is_fittable():
    """The preprocessor is a ColumnTransformer that fits and transforms."""
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)

    # Enough documents with shared vocabulary that the configured TF-IDF
    # ``min_df``/``max_df`` thresholds are satisfiable.
    emails = [
        Email(subject="project meeting", body="project review meeting tomorrow"),
        Email(subject="project update", body="project status meeting notes"),
        Email(subject="sale offer", body="discount offer free shipping now"),
        Email(subject="big sale", body="offer discount coupon free today"),
        Email(subject="team sync", body="meeting agenda for the project team"),
    ]
    frame = emails_to_frame(emails)
    matrix = pre.fit_transform(frame)
    # One row per email; columns = tfidf vocabulary + the numeric features.
    assert matrix.shape[0] == len(emails)
    assert matrix.shape[1] >= len(config.NUMERIC_FEATURES)
