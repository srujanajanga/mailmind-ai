"""Sentiment analysis for email text.

The primary engine is NLTK's VADER, which is well suited to the short,
punctuation-heavy register of email. VADER is loaded lazily inside a
``try``/``except`` so a machine without the lexicon (or without NLTK at all)
silently falls back to a small built-in polarity word list. Both paths emit a
:class:`~mailmind.schema.Sentiment` with a compound score in ``[-1, 1]``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..schema import Sentiment
from ..utils.text import tokenize

# VADER thresholds, per the canonical compound-score convention.
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05

# Compact fallback lexicon used only when VADER is unavailable.
_POSITIVE_WORDS: frozenset[str] = frozenset(
    """
    thanks thank thankful appreciate appreciated great good excellent awesome
    happy glad pleased love loved wonderful fantastic perfect congratulations
    congrats welcome nice helpful brilliant amazing delighted grateful success
    successful win won approved resolved fixed
    """.split()
)
_NEGATIVE_WORDS: frozenset[str] = frozenset(
    """
    bad terrible awful horrible hate hated angry upset disappointed disappointing
    sorry unfortunately problem issue issues fail failed failure broken bug error
    errors wrong delay delayed late missing complaint complaints refund cancel
    cancelled urgent worried concern concerned frustrated annoyed
    """.split()
)


@lru_cache(maxsize=1)
def _analyzer():
    """Return a cached VADER analyzer, or ``None`` if it cannot be loaded."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

        analyzer = SentimentIntensityAnalyzer()
        analyzer.polarity_scores("ok")  # force lexicon load up-front
        return analyzer
    except Exception:
        return None


def _label_for(compound: float) -> str:
    """Map a compound score to a positive/negative/neutral label."""
    if compound >= _POS_THRESHOLD:
        return "positive"
    if compound <= _NEG_THRESHOLD:
        return "negative"
    return "neutral"


def _fallback_score(text: str) -> float:
    """Lexicon-based compound estimate in ``[-1, 1]`` without VADER."""
    tokens = tokenize(text, lower=True)
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    if pos == neg:
        return 0.0
    # Normalise by the polarity-bearing tokens so a couple of strong words in a
    # short message register clearly, mirroring VADER's bounded output.
    return (pos - neg) / max(pos + neg, 1)


def analyze_sentiment(text: str) -> Sentiment:
    """Classify the sentiment of ``text``.

    Returns a :class:`~mailmind.schema.Sentiment` whose ``score`` is the
    compound polarity in ``[-1, 1]`` and whose ``confidence`` is its magnitude.
    Neutral results have a confidence near zero. Empty input is neutral.
    """
    text = (text or "").strip()
    if not text:
        return Sentiment(label="neutral", score=0.0, confidence=0.0)

    analyzer = _analyzer()
    if analyzer is not None:
        compound = float(analyzer.polarity_scores(text)["compound"])
    else:
        compound = _fallback_score(text)

    compound = max(-1.0, min(1.0, compound))
    return Sentiment(
        label=_label_for(compound),
        score=round(compound, 4),
        confidence=round(abs(compound), 4),
    )
