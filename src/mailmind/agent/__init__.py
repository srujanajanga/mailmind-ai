"""Agentic orchestration subsystem for MailMind AI.

Re-exports the high-level :class:`MailMindAgent` orchestrator together with the
two standalone helpers it relies on: :func:`summarize` (extractive summaries)
and :func:`suggest_actions` (the rule-based next-action recommender).
"""
from __future__ import annotations

from .actions import suggest_actions
from .agent import MailMindAgent
from .summarizer import summarize

__all__ = ["MailMindAgent", "summarize", "suggest_actions"]
