"""Rule-based urgency detection for email text.

Urgency is scored deterministically from three families of signals:

* an explicit lexicon of urgency phrases ("asap", "deadline", "action
  required", "final notice", ...),
* temporal pressure markers ("by 5pm", "today", "due tomorrow", "eod"), and
* stylistic intensity ("!!!", shouted ALL-CAPS words).

Matched cues are aggregated into a bounded ``0..1`` score and a human-readable
list of cues, then bucketed into high / medium / low. No optional dependency is
required, so the result is identical on every machine.
"""
from __future__ import annotations

import re
from typing import Optional

from ..schema import Email, Urgency
from ..schema import email_text as _email_text

# --------------------------------------------------------------------------- #
# Lexicon: phrase pattern -> (human-readable cue, weight)
# --------------------------------------------------------------------------- #
_PHRASE_CUES: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"\burgent(?:ly)?\b", re.I), "urgent", 0.45),
    (re.compile(r"\basap\b", re.I), "asap", 0.45),
    (re.compile(r"\bimmediately\b", re.I), "immediately", 0.40),
    (re.compile(r"\bright away\b", re.I), "right away", 0.35),
    (re.compile(r"\bemergency\b", re.I), "emergency", 0.50),
    (re.compile(r"\bcritical\b", re.I), "critical", 0.40),
    (re.compile(r"\baction required\b", re.I), "action required", 0.45),
    (re.compile(r"\bfinal notice\b", re.I), "final notice", 0.45),
    (re.compile(r"\blast chance\b", re.I), "last chance", 0.40),
    (re.compile(r"\btime[- ]sensitive\b", re.I), "time-sensitive", 0.40),
    (re.compile(r"\bdeadline\b", re.I), "deadline", 0.35),
    (re.compile(r"\bexpir(?:es|ing|e)\b", re.I), "expires", 0.30),
    (re.compile(r"\bdue (?:today|tomorrow|now)\b", re.I), "due soon", 0.40),
    (re.compile(r"\beod\b", re.I), "eod", 0.30),
    (re.compile(r"\b(?:respond|reply|rsvp)(?:\s+back)?\s+by\b", re.I), "reply by", 0.35),
    (re.compile(r"\bimportant\b", re.I), "important", 0.25),
    (re.compile(r"\bdon'?t (?:miss|wait)\b", re.I), "don't miss", 0.25),
    (re.compile(r"\bas soon as possible\b", re.I), "as soon as possible", 0.40),
]

# Temporal pressure markers (lower individual weight, additive).
_TIME_CUES: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"\bby \d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I), "by a set time", 0.25),
    (re.compile(r"\bby (?:today|tonight|tomorrow|noon|midnight)\b", re.I), "by a deadline", 0.25),
    (re.compile(r"\btoday\b", re.I), "today", 0.18),
    (re.compile(r"\btomorrow\b", re.I), "tomorrow", 0.15),
    (re.compile(r"\bwithin \d+\s*(?:hour|hr|minute|min)s?\b", re.I), "within a short window", 0.30),
]

_EXCLAIM_RE = re.compile(r"!")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")
# Common acronyms that are legitimately upper-case and should not count as
# "shouting" urgency.
_CAPS_ALLOWLIST: frozenset[str] = frozenset(
    {"FYI", "ASAP", "EOD", "RSVP", "USA", "URL", "PDF", "CEO", "CTO", "HR", "QA"}
)


def _style_cues(text: str) -> list[tuple[str, float]]:
    """Score stylistic intensity: excess exclamation marks and shouted words."""
    cues: list[tuple[str, float]] = []

    exclaims = len(_EXCLAIM_RE.findall(text))
    if exclaims >= 3:
        cues.append(("multiple exclamation marks", 0.25))
    elif exclaims >= 1:
        cues.append(("exclamation marks", 0.12))

    shouted = [w for w in _CAPS_WORD_RE.findall(text) if w not in _CAPS_ALLOWLIST]
    if len(shouted) >= 2:
        cues.append(("ALL-CAPS emphasis", 0.20))
    elif shouted:
        cues.append(("capitalised emphasis", 0.10))

    return cues


def _level_for(score: float) -> str:
    """Bucket a ``0..1`` urgency score into high / medium / low."""
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def detect_urgency(text: str, email: Optional["Email | dict | str"] = None) -> Urgency:
    """Score the urgency of ``text`` deterministically.

    ``email`` is accepted for interface symmetry; when provided as an
    :class:`~mailmind.schema.Email`/dict/str it is merged into the analysed text
    so subject and body are both considered. Returns an
    :class:`~mailmind.schema.Urgency` with a bounded score, a level, and the
    deduplicated list of matched cues.
    """
    text = text or ""
    if email is not None:
        extra = _email_text(email)
        if extra and extra not in text:
            text = f"{text} {extra}"
    if not text.strip():
        return Urgency(level="low", score=0.0, cues=[])

    matched: list[tuple[str, float]] = []
    for pattern, cue, weight in (*_PHRASE_CUES, *_TIME_CUES):
        if pattern.search(text):
            matched.append((cue, weight))
    matched.extend(_style_cues(text))

    # Aggregate with saturating diminishing returns so a pile-up of weak cues
    # cannot trivially max out the score while a couple of strong cues still
    # reads as clearly urgent.
    score = 0.0
    for _, weight in sorted(matched, key=lambda cw: -cw[1]):
        score += weight * (1.0 - score)
    score = max(0.0, min(1.0, score))

    cues: list[str] = []
    for cue, _ in matched:
        if cue not in cues:
            cues.append(cue)

    return Urgency(level=_level_for(score), score=round(score, 4), cues=cues)
