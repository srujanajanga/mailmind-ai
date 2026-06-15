"""MailMind AI — Your Inbox, Intelligently Organized.

An agentic AI email assistant that classifies, prioritises, summarises and
acts on email using classical NLP + machine learning.

The most-used objects are importable straight from the package root::

    from mailmind import MailMindAgent, MailMindClassifier, Email

Heavy sub-modules are loaded lazily via ``__getattr__`` so that simply importing
:mod:`mailmind` (e.g. to read :data:`mailmind.config.CATEGORIES`) never forces
scikit-learn, NLTK or the trained model into memory.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from mailmind import config  # noqa: F401  (cheap, no heavy deps)
from mailmind.schema import (  # noqa: F401
    Classification,
    Email,
    EmailInsight,
    Intent,
    NLPSignals,
    Priority,
    Sentiment,
    SuggestedAction,
    Urgency,
)

__version__ = "1.0.0"

# Map of public attribute -> "module:attr" resolved on first access.
_LAZY: dict[str, str] = {
    "MailMindAgent": "mailmind.agent.agent:MailMindAgent",
    "MailMindClassifier": "mailmind.ml.classifier:MailMindClassifier",
    "analyze_text": "mailmind.nlp:analyze_text",
    "Database": "mailmind.db.database:Database",
}

__all__ = [
    "Email",
    "EmailInsight",
    "Classification",
    "NLPSignals",
    "Sentiment",
    "Urgency",
    "Intent",
    "Priority",
    "SuggestedAction",
    "MailMindAgent",
    "MailMindClassifier",
    "analyze_text",
    "Database",
    "config",
    "__version__",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'mailmind' has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = import_module(module_path)
    return getattr(module, attr)
