"""Rule-based intent classification for email text.

Each email is scored against seven mutually-exclusive intents — ``request``,
``question``, ``meeting``, ``action_required``, ``fyi``, ``promotion`` and
``social`` — using regex cue families. The highest-scoring intent wins, with
``fyi`` as the neutral default when nothing matches. Confidence is derived from
the winning score relative to the total evidence, so a clean single-intent
match reads as confident while an ambiguous message does not. Fully
deterministic and dependency-free.
"""
from __future__ import annotations

import re

from ..schema import Intent

# --------------------------------------------------------------------------- #
# Cue families: intent -> list of (pattern, weight)
# --------------------------------------------------------------------------- #
_CUES: dict[str, list[tuple[re.Pattern[str], float]]] = {
    "meeting": [
        (re.compile(r"\b(?:meeting|meet up|catch up)\b", re.I), 1.0),
        (re.compile(r"\b(?:calendar|schedule|reschedule|availability)\b", re.I), 0.9),
        (re.compile(r"\b(?:zoom|google meet|teams|webex|hangout)\b", re.I), 0.9),
        (re.compile(r"\b(?:call|sync|standup|stand-up)\b", re.I), 0.6),
        (re.compile(r"\bat \d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I), 0.5),
        (re.compile(r"\b(?:invite|invitation)\b", re.I), 0.5),
    ],
    "action_required": [
        (re.compile(r"\baction required\b", re.I), 1.2),
        (re.compile(r"\b(?:verify|confirm|approve|authorise|authorize)\b", re.I), 0.9),
        (re.compile(r"\b(?:sign|signature|complete|submit)\b", re.I), 0.7),
        (re.compile(r"\b(?:pay|payment|invoice|overdue|outstanding)\b", re.I), 0.8),
        (re.compile(r"\b(?:reset|update) your (?:password|account|details)\b", re.I), 0.9),
        (re.compile(r"\brequired\b", re.I), 0.4),
    ],
    "promotion": [
        (re.compile(r"\b(?:sale|deal|deals|coupon|promo|promotion)\b", re.I), 1.0),
        (re.compile(r"\b(?:discount|save|offer|offers)\b", re.I), 0.8),
        (re.compile(r"\d+%\s*off\b", re.I), 1.0),
        (re.compile(r"\bunsubscribe\b", re.I), 0.9),
        (re.compile(r"\b(?:limited time|free shipping|buy now|shop now)\b", re.I), 0.8),
        (re.compile(r"\b(?:newsletter|exclusive)\b", re.I), 0.4),
    ],
    "social": [
        (re.compile(r"\b(?:liked|commented|reacted|mentioned|tagged)\b", re.I), 1.0),
        (re.compile(r"\b(?:follow|followed|friend request|connection request)\b", re.I), 0.9),
        (re.compile(r"\b(?:new (?:follower|connection)|added you)\b", re.I), 0.9),
        (re.compile(r"\b(?:notification|timeline|profile|post)\b", re.I), 0.4),
    ],
    "request": [
        (re.compile(r"\b(?:can|could|would) you\b", re.I), 0.9),
        (re.compile(r"\bplease\b", re.I), 0.6),
        (re.compile(r"\b(?:let me know|send me|share|provide|forward)\b", re.I), 0.7),
        (re.compile(r"\b(?:i need|we need|requesting|kindly)\b", re.I), 0.7),
    ],
    "question": [
        (re.compile(r"\?"), 0.8),
        (re.compile(r"\b(?:what|when|where|why|how|which|who)\b", re.I), 0.5),
        (re.compile(r"\b(?:do you|are you|is it|did you|have you)\b", re.I), 0.6),
        (re.compile(r"\b(?:any update|thoughts|wondering)\b", re.I), 0.5),
    ],
}

# Intents that ask the reader to *do* something out-rank passive ones on ties.
_PRIORITY: list[str] = [
    "action_required",
    "meeting",
    "request",
    "question",
    "promotion",
    "social",
]


def detect_intent(text: str) -> Intent:
    """Classify the dominant intent of ``text``.

    Returns an :class:`~mailmind.schema.Intent` whose ``label`` is one of
    ``request``/``question``/``meeting``/``action_required``/``fyi``/
    ``promotion``/``social`` and whose ``confidence`` reflects how decisively
    the winner out-scored competing intents. Empty or cue-free text is ``fyi``.
    """
    text = (text or "").strip()
    if not text:
        return Intent(label="fyi", confidence=0.0)

    scores: dict[str, float] = {}
    for intent, cues in _CUES.items():
        total = sum(weight for pattern, weight in cues if pattern.search(text))
        if total > 0.0:
            scores[intent] = total

    if not scores:
        return Intent(label="fyi", confidence=0.3)

    best = max(scores, key=lambda i: (scores[i], -_PRIORITY.index(i)))
    best_score = scores[best]
    total_score = sum(scores.values())

    # Confidence blends absolute evidence (is the best cue strong?) with
    # dominance (does it stand out from the rest?), bounded to [0, 1].
    strength = min(best_score / 1.2, 1.0)
    dominance = best_score / total_score
    confidence = max(0.0, min(1.0, 0.5 * strength + 0.5 * dominance))

    return Intent(label=best, confidence=round(confidence, 4))
