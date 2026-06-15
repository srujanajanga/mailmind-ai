"""The MailMind agent -- the orchestrator that ties every subsystem together.

:class:`MailMindAgent` takes a raw email, runs it through classification, NLP
enrichment, behavioural-aware priority scoring, extractive summarisation and the
rule-based action recommender, and returns a single :class:`EmailInsight`. It
also closes the learning loop by recording user feedback and exposes lightweight
aggregate statistics.

The agent is resilient by construction: every collaborator is optional and lazily
constructed, so an instance can be created (and individual emails processed) even
before a trained classifier model exists on disk.
"""
from __future__ import annotations

from typing import Any, Optional

from .. import config
from ..schema import Email, EmailInsight, as_email
from .actions import suggest_actions
from .summarizer import summarize

# Importance threshold above which a sender is flagged as a VIP.
_VIP_THRESHOLD = 0.8


class MailMindAgent:
    """End-to-end orchestrator for analysing and prioritising email.

    Parameters
    ----------
    classifier:
        A fitted classifier exposing ``classify``. When ``None`` the agent tries
        to load the persisted :class:`MailMindClassifier`, falling back to the
        rule-based :class:`HeuristicClassifier` if no model is available.
    db:
        Optional persistence layer. When supplied, processed emails and insights
        are saved and feedback is recorded.
    behavioral:
        Optional :class:`BehavioralLearner`; one is created over ``db`` if not
        given.
    scorer:
        Optional :class:`ContextScorer`; one is created over ``behavioral`` if
        not given.
    model_path:
        Filesystem location of the persisted classifier model.
    """

    def __init__(
        self,
        classifier: Optional[Any] = None,
        db: Optional[Any] = None,
        behavioral: Optional[Any] = None,
        scorer: Optional[Any] = None,
        model_path: Any = config.MODEL_PATH,
    ) -> None:
        self.db = db
        self.classifier = classifier or self._load_classifier(model_path)
        self.behavioral = behavioral or self._build_behavioral(db)
        self.scorer = scorer or self._build_scorer(self.behavioral)

    # ------------------------------------------------------------------ #
    # Lazy collaborator construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_classifier(model_path: Any) -> Any:
        """Load the trained classifier, falling back to the heuristic one.

        Imports are performed lazily so the agent module stays importable even
        while sibling subsystems are still being written.
        """
        from ..ml.classifier import HeuristicClassifier, MailMindClassifier

        try:
            return MailMindClassifier.load(model_path)
        except Exception:
            # Missing/corrupt model, or ML stack unavailable: degrade to rules.
            return HeuristicClassifier()

    @staticmethod
    def _build_behavioral(db: Optional[Any]) -> Any:
        """Construct a :class:`BehavioralLearner` bound to ``db``."""
        from ..behavioral.learner import BehavioralLearner

        return BehavioralLearner(db)

    @staticmethod
    def _build_scorer(behavioral: Any) -> Any:
        """Construct a :class:`ContextScorer` over ``behavioral``."""
        from ..context.scorer import ContextScorer

        return ContextScorer(behavioral)

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #
    def process_email(self, email: "Email | dict | str") -> EmailInsight:
        """Run the full analysis pipeline over a single email.

        Returns a fully-populated :class:`EmailInsight`. When a database is
        configured, the email and its insight are persisted as a side effect.
        """
        from .. import nlp as nlp_module

        mail = as_email(email)

        classification = self.classifier.classify(mail)
        nlp = nlp_module.analyze_text(mail.text, mail)
        priority = self.scorer.score(mail, classification, nlp)
        summary = summarize(mail.body or mail.text)
        actions = suggest_actions(mail, classification, nlp, priority)
        flags = self._flags(mail, classification, nlp, priority)

        insight = EmailInsight(
            email=mail,
            classification=classification,
            nlp=nlp,
            priority=priority,
            summary=summary,
            suggested_actions=actions,
            flags=flags,
        )

        self._persist(mail, insight)
        return insight

    def process_inbox(self, emails: list["Email | dict | str"]) -> list[EmailInsight]:
        """Analyse a batch of emails, returned sorted by priority (desc)."""
        insights = [self.process_email(email) for email in emails]
        insights.sort(key=lambda insight: insight.priority.score, reverse=True)
        return insights

    def _flags(
        self,
        email: Email,
        classification: Any,
        nlp: Any,
        priority: Any,
    ) -> list[str]:
        """Derive the boolean attention flags attached to an insight."""
        flags: list[str] = []

        if nlp.urgency.level == "high" or priority.band == "Critical":
            flags.append("urgent")
        if self._sender_importance(email) >= _VIP_THRESHOLD:
            flags.append("vip")
        if classification.label == "Spam":
            flags.append("spam")
        if classification.label == "Promotions":
            flags.append("promo")

        return flags

    def _sender_importance(self, email: Email) -> float:
        """Sender importance via the scorer, defaulting to 0.0 on any failure."""
        try:
            return float(self.scorer.sender_importance(email))
        except Exception:
            return 0.0

    def _persist(self, email: Email, insight: EmailInsight) -> None:
        """Best-effort persistence of an email and its insight."""
        if self.db is None:
            return
        self.db.save_email(email)
        self.db.save_insight(insight)

    # ------------------------------------------------------------------ #
    # Feedback + stats
    # ------------------------------------------------------------------ #
    def record_feedback(self, email: "Email | dict | str", action: str) -> None:
        """Record a user action so behaviour and persistence stay in sync.

        Delegated to :class:`BehavioralLearner`, which updates in-memory tallies
        and -- when a database is attached -- the persisted action log too.
        """
        self.behavioral.record_action(email, action)

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about processed mail and feedback.

        Database-backed counts are included when available; the response always
        names the active classifier so callers can tell whether the trained
        model or the heuristic fallback is in use.
        """
        result: dict[str, Any] = {
            "actions": {},
            "total_emails": 0,
            "classifier": type(self.classifier).__name__,
        }
        if self.db is not None:
            try:
                result["actions"] = self.db.action_counts()
                result["total_emails"] = self.db.total_emails()
            except Exception:
                # Leave the defaults in place if the db query fails.
                pass
        return result
