"""Behavioural-learning subsystem for MailMind AI.

Re-exports :class:`BehavioralLearner`, which personalises priority scoring by
learning from the actions a user takes on their mail.
"""
from __future__ import annotations

from .learner import BehavioralLearner

__all__ = ["BehavioralLearner"]
