"""Behavioural learning from observed user actions.

The :class:`BehavioralLearner` watches how the user interacts with mail
(``replied`` / ``opened`` / ``ignored`` / ``deleted``) and turns those signals
into a continuous *engagement* score for each sender and category. Downstream the
context scorer uses that engagement to personalise priority: mail from senders
the user reliably replies to floats up, mail they habitually delete sinks.

Persistence is optional. When constructed with a :class:`mailmind.db.database.Database`
the counts live in SQLite; otherwise identical-shaped in-memory dictionaries are
used so the learner is fully functional without a backing store.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from .. import config
from ..schema import Email, as_email


class BehavioralLearner:
    """Learn per-sender and per-category engagement from user actions.

    Parameters
    ----------
    db:
        Optional database exposing ``record_action``, ``sender_action_counts``
        and ``category_action_counts``. When ``None`` the learner keeps its own
        in-memory action tallies of the same shape.
    """

    def __init__(self, db: Optional[object] = None) -> None:
        self._db = db
        # In-memory fall-backs: mapping name -> {action: count}.
        self._sender_counts: dict[str, dict[str, int]] = defaultdict(
            self._empty_counts
        )
        self._category_counts: dict[str, dict[str, int]] = defaultdict(
            self._empty_counts
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _empty_counts() -> dict[str, int]:
        """Return a zeroed count dict covering every valid action."""
        return {action: 0 for action in config.VALID_ACTIONS}

    @staticmethod
    def _engagement_from_counts(counts: dict[str, int]) -> float:
        """Squash weighted action counts into an engagement score in ``[-1, 1]``.

        Each action contributes its :data:`config.ACTION_WEIGHTS` value times the
        number of times it occurred. The weighted sum is normalised by the total
        action volume (plus one, to damp tiny samples) and passed through ``tanh``
        so the result stays bounded and saturates gracefully.
        """
        total = sum(max(0, counts.get(a, 0)) for a in config.VALID_ACTIONS)
        if total == 0:
            return 0.0
        weighted = sum(
            config.ACTION_WEIGHTS[a] * max(0, counts.get(a, 0))
            for a in config.VALID_ACTIONS
        )
        return float(math.tanh(weighted / (total + 1)))

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_action(self, email: "Email | dict | str", action: str) -> None:
        """Record that the user performed ``action`` on ``email``.

        Unknown actions (not in :data:`config.VALID_ACTIONS`) are ignored. Both
        the sender tally and the category tally are updated; the category update
        is skipped when the email carries no ground-truth ``label``.
        """
        if action not in config.VALID_ACTIONS:
            return

        email = as_email(email)
        sender = email.sender or ""
        category = email.label

        if self._db is not None:
            self._db.record_action(
                email_id=email.id,
                sender=sender,
                category=category or "",
                action=action,
                ts=email.timestamp,
            )
            return

        self._sender_counts[sender][action] += 1
        if category:
            self._category_counts[category][action] += 1

    # ------------------------------------------------------------------ #
    # Engagement
    # ------------------------------------------------------------------ #
    def sender_engagement(self, sender: str) -> float:
        """Return engagement with ``sender`` in ``[-1, 1]`` (0.0 if unseen)."""
        if not sender:
            return 0.0
        counts = self._sender_action_counts(sender)
        return self._engagement_from_counts(counts)

    def category_engagement(self, category: str) -> float:
        """Return engagement with ``category`` in ``[-1, 1]`` (0.0 if unseen)."""
        if not category:
            return 0.0
        counts = self._category_action_counts(category)
        return self._engagement_from_counts(counts)

    def _sender_action_counts(self, sender: str) -> dict[str, int]:
        """Fetch raw per-action counts for a sender from db or memory."""
        if self._db is not None:
            return self._db.sender_action_counts(sender)
        return self._sender_counts.get(sender, self._empty_counts())

    def _category_action_counts(self, category: str) -> dict[str, int]:
        """Fetch raw per-action counts for a category from db or memory."""
        if self._db is not None:
            return self._db.category_action_counts(category)
        return self._category_counts.get(category, self._empty_counts())

    # ------------------------------------------------------------------ #
    # Score adjustment
    # ------------------------------------------------------------------ #
    def adjust(
        self,
        base_score_0_1: float,
        email: "Email | dict | str",
        category: Optional[str],
    ) -> tuple[float, list[str]]:
        """Nudge a base priority score by learned engagement.

        Parameters
        ----------
        base_score_0_1:
            The unadjusted priority signal in ``[0, 1]``.
        email:
            The email being scored (used for its sender).
        category:
            The predicted/known category to blend in. May be ``None``.

        Returns
        -------
        tuple
            ``(adjusted_score, reasons)`` where ``adjusted_score`` stays in
            ``[0, 1]`` and ``reasons`` is a list of human-readable explanations
            (empty when engagement is neutral).
        """
        email = as_email(email)
        sender = email.sender or ""

        sender_eng = self.sender_engagement(sender)
        category_eng = self.category_engagement(category or "")
        engagement = 0.6 * sender_eng + 0.4 * category_eng

        adjusted = base_score_0_1 + 0.2 * engagement
        adjusted = float(min(1.0, max(0.0, adjusted)))

        reasons = self._explain(sender_eng, category_eng, category)
        return adjusted, reasons

    @staticmethod
    def _explain(
        sender_eng: float, category_eng: float, category: Optional[str]
    ) -> list[str]:
        """Translate engagement signals into human-readable reasons."""
        reasons: list[str] = []

        if sender_eng > 0.3:
            reasons.append("You frequently reply to this sender")
        elif sender_eng < -0.3:
            reasons.append("You usually ignore this sender")

        if category:
            if category_eng > 0.3:
                reasons.append(f"You engage often with {category} mail")
            elif category_eng < -0.3:
                reasons.append(f"You usually skip {category} mail")

        return reasons
