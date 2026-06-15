"""Tests for the NLP layer (``mailmind.nlp``)."""
from __future__ import annotations

from mailmind import nlp
from mailmind.schema import NLPSignals
from mailmind.utils.text import get_stopwords


def test_analyze_sentiment_positive_vs_negative():
    """Positive text scores above neutral; negative text scores below."""
    positive = nlp.analyze_sentiment(
        "Thank you so much, this is wonderful and I really appreciate your help!"
    )
    negative = nlp.analyze_sentiment(
        "This is terrible, I am very disappointed and angry about the broken service."
    )
    assert positive.label == "positive"
    assert positive.score > 0.0
    assert negative.label == "negative"
    assert negative.score < 0.0


def test_detect_urgency_high():
    """An obviously urgent message is flagged high."""
    urgency = nlp.detect_urgency("URGENT reply ASAP deadline today!!!")
    assert urgency.level == "high"
    assert urgency.score >= 0.6
    assert urgency.cues  # at least one cue identified


def test_detect_urgency_low():
    """A calm message reads as low urgency."""
    urgency = nlp.detect_urgency("Just sharing some photos from the weekend trip.")
    assert urgency.level == "low"


def test_detect_intent_variants():
    """Intent detection picks the right dominant intent for clear cases."""
    assert nlp.detect_intent("Can we schedule a meeting on Zoom tomorrow?").label == "meeting"
    assert nlp.detect_intent("What time does the store open?").label == "question"
    assert (
        nlp.detect_intent("Huge sale! 50% off, limited time offer, shop now.").label
        == "promotion"
    )


def test_extract_keywords_respects_top_k_and_stopwords():
    """Keyword extraction honours top_k and excludes stop-words."""
    text = (
        "The quarterly project review meeting will cover the budget report and "
        "the client deliverable schedule for the team."
    )
    keywords = nlp.extract_keywords(text, top_k=5)
    assert len(keywords) <= 5

    stops = get_stopwords()
    # No single-word keyword is a stop-word.
    assert all(not (len(kw.split()) == 1 and kw in stops) for kw in keywords)


def test_analyze_text_bundles_all_signals():
    """analyze_text returns a fully-populated NLPSignals object."""
    signals = nlp.analyze_text("URGENT: please verify your account immediately!!!")
    assert isinstance(signals, NLPSignals)
    assert isinstance(signals.keywords, list)
    assert signals.urgency.level in {"low", "medium", "high"}
    assert signals.sentiment.label in {"positive", "neutral", "negative"}
    assert signals.intent.label  # non-empty intent label
