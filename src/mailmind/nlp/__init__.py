"""MailMind AI natural-language understanding layer.

This package turns raw email text into structured signals consumed by the
classifier, the priority scorer and the agent. Each analysis is independent and
deterministic; :func:`analyze_text` bundles all four into a single
:class:`~mailmind.schema.NLPSignals` object.

Public functions
----------------
* :func:`extract_keywords` -- salient keywords / phrases.
* :func:`analyze_sentiment` -- VADER-based polarity (with fallback).
* :func:`detect_urgency` -- rule-based urgency scoring.
* :func:`detect_intent` -- rule-based intent classification.
* :func:`analyze_text` -- all of the above, combined.
"""
from __future__ import annotations

from typing import Optional

from ..schema import Email, NLPSignals
from .intent import detect_intent
from .keywords import extract_keywords
from .sentiment import analyze_sentiment
from .urgency import detect_urgency

__all__ = [
    "extract_keywords",
    "analyze_sentiment",
    "detect_urgency",
    "detect_intent",
    "analyze_text",
]


def analyze_text(
    text: str, email: Optional["Email | dict | str"] = None
) -> NLPSignals:
    """Run the full NLP pipeline over ``text`` and bundle the results.

    Parameters
    ----------
    text:
        The email text to analyse (typically ``Email.text``).
    email:
        Optional originating email; forwarded to :func:`detect_urgency` so that
        structural signals (e.g. subject text) can reinforce the urgency score.

    Returns
    -------
    NLPSignals
        Container holding keywords, sentiment, urgency and intent.
    """
    return NLPSignals(
        keywords=extract_keywords(text),
        sentiment=analyze_sentiment(text),
        urgency=detect_urgency(text, email),
        intent=detect_intent(text),
    )
