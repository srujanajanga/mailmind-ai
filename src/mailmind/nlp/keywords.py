"""Deterministic keyword extraction for email text.

The primary path scores candidate terms by term-frequency weighted by their
earliest position in the document (subjects and opening sentences carry the
most signal in email).  When spaCy *and* an English model are installed, noun
chunks are mixed in so multi-word phrases ("payment deadline", "project
review") surface alongside salient unigrams.  Everything degrades gracefully:
without spaCy the extractor uses unigram + bigram frequencies and never touches
an optional dependency outside a ``try``/``except`` guard.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

from ..utils.text import content_tokens, get_stopwords, normalize_whitespace


# --------------------------------------------------------------------------- #
# Optional spaCy wiring (best-effort, fully cached, never required)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _spacy_nlp():
    """Return a loaded spaCy English pipeline, or ``None`` if unavailable.

    Loading is attempted exactly once and cached; any failure (missing
    package, missing model, import error) yields ``None`` so callers fall back
    to the frequency-based path.
    """
    try:
        import spacy  # type: ignore

        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _position_weight(index: int) -> float:
    """Down-weight terms appearing later in the document.

    The opening tokens of an email (subject + first line) are the most
    informative, so the weight decays smoothly with the first-seen position.
    """
    return 1.0 + math.log1p(1.0 / (1.0 + index))


def _frequency_scores(tokens: list[str]) -> dict[str, float]:
    """Score unigrams (and repeated bigrams) by weighted frequency.

    Unigrams form the backbone of the ranking. A bigram is only admitted when
    it recurs (frequency >= 2): repeated adjacent pairs are genuine phrases
    ("project review", "due date"), whereas one-off pairs are just incidental
    word order and would otherwise flood the result with overlapping chains.
    """
    first_seen: dict[str, int] = {}
    uni_counts: dict[str, int] = {}
    bi_counts: dict[str, int] = {}

    # Unigrams: tokens are already stop-word-filtered by ``content_tokens``.
    for position, token in enumerate(tokens):
        first_seen.setdefault(token, position)
        uni_counts[token] = uni_counts.get(token, 0) + 1

    # Candidate bigrams from adjacent content tokens.
    for position in range(len(tokens) - 1):
        phrase = f"{tokens[position]} {tokens[position + 1]}"
        first_seen.setdefault(phrase, position)
        bi_counts[phrase] = bi_counts.get(phrase, 0) + 1

    scores: dict[str, float] = {
        term: count * _position_weight(first_seen[term])
        for term, count in uni_counts.items()
    }
    for phrase, count in bi_counts.items():
        if count >= 2:  # keep only recurring, meaningful phrases
            scores[phrase] = count * 1.5 * _position_weight(first_seen[phrase])
    return scores


def _spacy_phrase_scores(text: str, nlp) -> dict[str, float]:
    """Score spaCy noun chunks plus salient nouns/proper nouns by position."""
    stops = get_stopwords()
    doc = nlp(text)
    n_tokens = max(len(doc), 1)
    scores: dict[str, float] = {}

    def _add(term: str, position: int, boost: float) -> None:
        term = term.strip().lower()
        if len(term) < 3 or all(w in stops for w in term.split()):
            return
        weight = boost * _position_weight(int(position / n_tokens * 10))
        scores[term] = max(scores.get(term, 0.0), weight)

    for chunk in doc.noun_chunks:
        words = [t.text.lower() for t in chunk if not t.is_stop and t.is_alpha]
        phrase = " ".join(words)
        if phrase:
            _add(phrase, chunk.start, boost=1.6 if len(words) > 1 else 1.0)

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha and not token.is_stop:
            _add(token.text.lower(), token.i, boost=1.0)

    return scores


def _dedupe(ordered: list[str]) -> list[str]:
    """Drop phrases fully subsumed by an already-selected phrase, keep order."""
    kept: list[str] = []
    seen_words: set[str] = set()
    for term in ordered:
        words = term.split()
        if len(words) == 1 and term in seen_words:
            continue
        kept.append(term)
        seen_words.update(words)
    return kept


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_keywords(text: str, top_k: int = 8) -> list[str]:
    """Return up to ``top_k`` salient, lower-cased keywords/phrases.

    Results are deterministic: ties are broken by alphabetical order so the
    same input always yields the same output. When spaCy is installed the
    extractor prefers multi-word noun phrases; otherwise it ranks unigrams and
    bigrams by frequency weighted by first-seen position.
    """
    text = normalize_whitespace(text or "")
    if not text or top_k <= 0:
        return []

    nlp = _spacy_nlp()
    if nlp is not None:
        scores: dict[str, float] = _spacy_phrase_scores(text, nlp)
        if not scores:  # spaCy produced nothing usable; fall back.
            scores = _frequency_scores(content_tokens(text))
    else:
        scores = _frequency_scores(content_tokens(text))

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return _dedupe([term for term, _ in ranked])[:top_k]
