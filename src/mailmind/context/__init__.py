"""Context-aware priority scoring subsystem for MailMind AI.

Exposes :class:`ContextScorer`, which blends category importance, urgency,
sender importance, learned behaviour and message freshness into a single 0-100
priority score with a human-readable explanation.
"""
from __future__ import annotations

from .scorer import ContextScorer

__all__ = ["ContextScorer"]
