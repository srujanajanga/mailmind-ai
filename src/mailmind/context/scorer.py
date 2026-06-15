"""Context-aware priority scoring for MailMind AI.

The :class:`ContextScorer` fuses several normalised (0-1) signals into a single
0-100 priority score: how important the predicted *category* is, how *urgent*
the message reads, how important the *sender* is (VIP rules + learned
behaviour), the user's learned *engagement*, and message *freshness*. The blend
is governed by :data:`mailmind.config.PRIORITY_WEIGHTS`, and any optional
:class:`~mailmind.behavioral.learner.BehavioralLearner` gets the final say via
its ``adjust`` hook so the score reflects how the user actually treats similar
mail.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .. import config
from ..schema import Classification, Email, NLPSignals, Priority, as_email

# Deterministic "now" anchor so scoring is reproducible against the static
# synthetic dataset (whose timestamps cluster around mid-June 2026).
_NOW_ANCHOR = datetime(2026, 6, 14, 9, 0, 0)

# Freshness decays to ~0 over roughly one week.
_FRESHNESS_DECAY_DAYS = 7.0

# How strongly a matched VIP keyword lifts an otherwise unknown sender.
_VIP_KEYWORD_BOOST = 0.4


class ContextScorer:
    """Combine category, urgency, sender, behaviour and freshness into a score.

    Parameters
    ----------
    behavioral:
        Optional behavioural learner. When supplied it contributes the
        ``behavior`` component, sharpens ``sender_importance`` and gets the
        final adjustment pass; when ``None`` the scorer falls back to neutral
        defaults so it works fully standalone.
    vip_senders:
        Optional iterable of explicit VIP sender addresses or domains. These are
        merged with :data:`mailmind.config.VIP_DOMAINS`.
    """

    def __init__(
        self,
        behavioral: Optional[Any] = None,
        vip_senders: Optional[Iterable[str]] = None,
    ) -> None:
        self.behavioral = behavioral
        explicit = {str(s).lower() for s in vip_senders} if vip_senders else set()
        self.vip_senders: set[str] = explicit | {d.lower() for d in config.VIP_DOMAINS}
        self.vip_keywords: set[str] = {k.lower() for k in config.VIP_KEYWORDS}

    # ------------------------------------------------------------------ #
    # Sender importance
    # ------------------------------------------------------------------ #
    def sender_importance(self, email: "Email | dict | str") -> float:
        """Return a 0-1 importance score for the email's sender.

        A direct VIP address/domain match scores ``1.0``. A VIP keyword found in
        the sender name/address adds a boost. When a behavioural learner is
        present its sender engagement (mapped from ``[-1, 1]`` to ``[0, 1]``) is
        blended in so habitual correspondents float up over time.
        """
        mail = as_email(email)
        sender = (mail.sender or "").lower()
        domain = (mail.sender_domain or "").lower()
        name = (mail.sender_name or "").lower()

        if sender in self.vip_senders or domain in self.vip_senders:
            return 1.0

        score = 0.0
        haystack = f"{name} {sender}"
        if any(keyword in haystack for keyword in self.vip_keywords):
            score += _VIP_KEYWORD_BOOST

        if self.behavioral is not None:
            engagement = self._safe_engagement(self.behavioral.sender_engagement, sender)
            # Blend: take the stronger of the keyword signal and learned
            # engagement so a well-engaged sender is not dragged down by the
            # absence of a keyword, and vice-versa.
            score = max(score, _remap_unit(engagement))

        return _clamp(score)

    # ------------------------------------------------------------------ #
    # Full priority score
    # ------------------------------------------------------------------ #
    def score(
        self,
        email: "Email | dict | str",
        classification: Classification,
        nlp: NLPSignals,
    ) -> Priority:
        """Compute the weighted 0-100 :class:`~mailmind.schema.Priority`."""
        mail = as_email(email)

        category_c = config.CATEGORY_PRIORITY.get(classification.label, 0.5)
        urgency_c = _clamp(_urgency_score(nlp))
        sender_c = self.sender_importance(mail)
        behavior_c = self._behavior_component(mail, classification.label)
        freshness_c = self._freshness(mail.timestamp)

        components = {
            "category": category_c,
            "urgency": urgency_c,
            "sender": sender_c,
            "behavior": behavior_c,
            "freshness": freshness_c,
        }
        weighted = sum(
            config.PRIORITY_WEIGHTS[key] * value for key, value in components.items()
        )
        weighted = _clamp(weighted)

        # Let the behavioural learner nudge the blended score and explain why.
        extra_reasons: list[str] = []
        if self.behavioral is not None:
            try:
                weighted, extra_reasons = self.behavioral.adjust(
                    weighted, mail, classification.label
                )
                weighted = _clamp(float(weighted))
            except Exception:  # noqa: BLE001 - behaviour layer must never crash scoring
                extra_reasons = []

        score = round(weighted * 100.0, 1)
        band = config.band_for_score(score)
        reasons = self._reasons(
            mail, classification, nlp, sender_c, freshness_c, extra_reasons
        )
        return Priority(score=score, band=band, reasons=reasons)

    # ------------------------------------------------------------------ #
    # Component helpers
    # ------------------------------------------------------------------ #
    def _behavior_component(self, email: Email, category: str) -> float:
        """Map learned engagement to a 0-1 component (neutral 0.5 if unknown)."""
        if self.behavioral is None:
            return 0.5
        sender = (email.sender or "").lower()
        sender_eng = self._safe_engagement(self.behavioral.sender_engagement, sender)
        category_eng = self._safe_engagement(
            self.behavioral.category_engagement, category
        )
        # Sender behaviour dominates; category behaviour is a softer prior.
        blended = 0.7 * sender_eng + 0.3 * category_eng
        return _clamp(_remap_unit(blended))

    def _freshness(self, timestamp: str) -> float:
        """Linear recency in ``[0, 1]`` decaying over ~7 days (0.5 if unknown)."""
        parsed = _parse_timestamp(timestamp)
        if parsed is None:
            return 0.5
        age_days = (_NOW_ANCHOR - parsed).total_seconds() / 86400.0
        if age_days <= 0.0:               # future timestamp -> treat as brand new
            return 1.0
        if age_days >= _FRESHNESS_DECAY_DAYS:
            return 0.0
        return 1.0 - (age_days / _FRESHNESS_DECAY_DAYS)

    def _reasons(
        self,
        email: Email,
        classification: Classification,
        nlp: NLPSignals,
        sender_c: float,
        freshness_c: float,
        extra_reasons: list[str],
    ) -> list[str]:
        """Build a short, human-readable explanation of the score."""
        reasons: list[str] = []

        category_c = config.CATEGORY_PRIORITY.get(classification.label, 0.5)
        if category_c >= 0.7:
            reasons.append(f"High-priority category: {classification.label}")

        urgency = getattr(nlp, "urgency", None)
        if urgency is not None and getattr(urgency, "level", "low") != "low":
            cues = getattr(urgency, "cues", None) or []
            if cues:
                reasons.append(f"Urgency cues detected: {cues[0]}")
            else:
                reasons.append("Message reads as urgent")

        if sender_c >= 0.99:
            reasons.append("VIP sender")
        elif sender_c >= 0.6:
            reasons.append("Sender frequently engaged")

        if freshness_c >= 0.85:
            reasons.append("Recent message")

        reasons.extend(r for r in extra_reasons if r)

        # De-duplicate while preserving order, then keep the top few.
        seen: set[str] = set()
        unique = [r for r in reasons if not (r in seen or seen.add(r))]
        return unique[:4]

    # ------------------------------------------------------------------ #
    # Behavioural-layer guards
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_engagement(fn: Any, key: str) -> float:
        """Call a behavioural engagement function, defaulting to 0 on failure."""
        try:
            return _clamp(float(fn(key)), -1.0, 1.0)
        except Exception:  # noqa: BLE001 - never let the behaviour layer break scoring
            return 0.0


# --------------------------------------------------------------------------- #
# Module-level numeric helpers
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _remap_unit(value: float) -> float:
    """Map a value from ``[-1, 1]`` onto ``[0, 1]``."""
    return (_clamp(value, -1.0, 1.0) + 1.0) / 2.0


def _urgency_score(nlp: NLPSignals) -> float:
    """Pull the urgency score out of an NLP signal bundle, defensively."""
    urgency = getattr(nlp, "urgency", None)
    if urgency is None:
        return 0.0
    try:
        return float(getattr(urgency, "score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _parse_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into a naive (UTC-normalised) datetime.

    Returns ``None`` when the value is missing or unparseable so callers can
    fall back to a neutral freshness score.
    """
    if not timestamp:
        return None
    text = str(timestamp).strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed
