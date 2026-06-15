"""Feature engineering for the MailMind classifier.

This module is the *single source of truth* for the engineered numeric features
the model consumes. Everything downstream (training, evaluation, inference)
builds its design matrix through :func:`emails_to_frame` and
:func:`build_preprocessor`, guaranteeing that the same transformations are
applied at train and predict time.

The 8 engineered features mirror :data:`mailmind.config.NUMERIC_FEATURES` and
capture cheap, interpretable signals (length, links, shouting, urgency / money
cues) that complement the TF-IDF bag-of-words representation.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from mailmind import config
from mailmind.schema import Email, as_email
from mailmind.utils.text import clean_text, count_links, uppercase_ratio

# --------------------------------------------------------------------------- #
# Lexicons for the engineered cue-count features
# --------------------------------------------------------------------------- #
#: Words/phrases that signal a message wants the reader to act *now*.
URGENCY_LEXICON: tuple[str, ...] = (
    "urgent",
    "asap",
    "immediately",
    "deadline",
    "important",
    "action required",
    "now",
    "today",
    "expires",
    "final notice",
    "verify",
    "suspended",
)

#: Tokens/symbols that signal promotional or money-related content.
MONEY_LEXICON: tuple[str, ...] = (
    "$",
    "%",
    "free",
    "win",
    "prize",
    "cash",
    "offer",
    "discount",
    "sale",
)


def _count_occurrences(haystack: str, needles: Iterable[str]) -> int:
    """Return the total number of (overlapping-free) needle occurrences."""
    return sum(haystack.count(needle) for needle in needles)


def compute_engineered_features(email: "Email | dict | str") -> dict[str, float]:
    """Compute the engineered numeric features for one email.

    Args:
        email: An :class:`~mailmind.schema.Email`, a dict, or raw text. Any of
            these is coerced via :func:`~mailmind.schema.as_email`.

    Returns:
        A dict whose keys are exactly :data:`mailmind.config.NUMERIC_FEATURES`.
    """
    em = as_email(email)
    subject = em.subject or ""
    body = em.body or ""
    text = em.text
    lowered = text.lower()

    links = em.num_links if em.num_links else count_links(text)

    return {
        "body_length": float(len(body.split())),
        "subject_length": float(len(subject.split())),
        "num_links": float(links),
        "has_attachment": float(int(bool(em.has_attachment))),
        "exclaim_count": float(text.count("!")),
        "uppercase_ratio": float(uppercase_ratio(f"{subject} {body}")),
        "urgency_hits": float(_count_occurrences(lowered, URGENCY_LEXICON)),
        "money_hits": float(_count_occurrences(lowered, MONEY_LEXICON)),
    }


def emails_to_frame(emails: "Iterable[Email | dict | str]") -> pd.DataFrame:
    """Vectorise a collection of emails into a model-ready DataFrame.

    Args:
        emails: An iterable of :class:`~mailmind.schema.Email`, dicts, or
            strings (mixed types are allowed).

    Returns:
        A :class:`pandas.DataFrame` with columns ``["clean_text",
        *config.NUMERIC_FEATURES]`` — exactly what :func:`build_preprocessor`
        expects.
    """
    rows: list[dict[str, float | str]] = []
    for raw in emails:
        em = as_email(raw)
        row: dict[str, float | str] = {"clean_text": clean_text(em.text)}
        row.update(compute_engineered_features(em))
        rows.append(row)

    columns = ["clean_text", *config.NUMERIC_FEATURES]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_preprocessor() -> ColumnTransformer:
    """Build the feature pipeline shared by every candidate estimator.

    The transformer applies a :class:`TfidfVectorizer` to the ``clean_text``
    column and a :class:`MaxAbsScaler` to the engineered numeric columns. Both
    keep the output non-negative (required by ``ComplementNB``) and sparse.

    Returns:
        A fitted-on-demand :class:`sklearn.compose.ColumnTransformer`.
    """
    return ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS), "clean_text"),
            ("num", MaxAbsScaler(), list(config.NUMERIC_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
