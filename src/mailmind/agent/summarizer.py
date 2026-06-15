"""Dependency-light extractive summariser for MailMind AI.

The agent uses :func:`summarize` to turn an email body into a one- or two-line
gist. The approach is a classic frequency-based extractive summary: score each
sentence by the summed frequency of the content words it contains, keep the
top-N highest-scoring sentences, and emit them in their original reading order.

This deliberately avoids any heavyweight model so it runs with nothing more than
the standard library plus the project's NLTK-backed text utilities.
"""
from __future__ import annotations

from collections import Counter

from ..utils.text import content_tokens, normalize_whitespace, sentences


def summarize(text: str, max_sentences: int = 2) -> str:
    """Return an extractive summary of ``text`` of at most ``max_sentences``.

    Sentences are scored by the summed corpus frequency of their content words
    (normalised by sentence length so long sentences are not unfairly favoured),
    and the best ``max_sentences`` are returned in their original order joined by
    a single space. When the text already has ``max_sentences`` sentences or
    fewer, the whitespace-normalised text is returned unchanged.

    Parameters
    ----------
    text:
        The raw text to condense (typically an email body).
    max_sentences:
        Upper bound on the number of sentences in the summary. Values below one
        are treated as one.

    Returns
    -------
    str
        The extractive summary, or an empty string for empty input.
    """
    cleaned = normalize_whitespace(text or "")
    if not cleaned:
        return ""

    max_sentences = max(1, max_sentences)

    sents = sentences(cleaned)
    if len(sents) <= max_sentences:
        return cleaned

    # Build a document-level frequency table over content words.
    frequencies: Counter[str] = Counter()
    for sentence in sents:
        frequencies.update(content_tokens(sentence))

    if not frequencies:
        # No scorable content words (e.g. all stopwords/punctuation): fall back
        # to the leading sentences, which carry the most context in email prose.
        return " ".join(sents[:max_sentences])

    peak = max(frequencies.values())

    def score(sentence: str) -> float:
        """Mean normalised content-word weight for a single sentence."""
        tokens = content_tokens(sentence)
        if not tokens:
            return 0.0
        return sum(frequencies[token] / peak for token in tokens) / len(tokens)

    ranked = sorted(
        range(len(sents)),
        key=lambda i: score(sents[i]),
        reverse=True,
    )
    chosen = sorted(ranked[:max_sentences])
    return " ".join(sents[i] for i in chosen)
