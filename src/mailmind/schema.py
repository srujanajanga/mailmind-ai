"""Core data contracts shared across every MailMind AI module.

These light-weight ``dataclass`` objects are the lingua franca of the system:
the dataset generator emits :class:`Email` rows, the NLP/ML layers enrich them,
and the agent returns an :class:`EmailInsight` for each message.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Email
# --------------------------------------------------------------------------- #
@dataclass
class Email:
    """A single email message.

    Only ``subject`` and ``body`` are strictly required; everything else has a
    sensible default so callers (and the REST API) can build partial objects.
    """

    subject: str = ""
    body: str = ""
    sender: str = ""                 # e.g. "alice@work.com"
    sender_name: str = ""            # e.g. "Alice Smith"
    sender_domain: str = ""          # e.g. "work.com"
    recipient: str = ""
    timestamp: str = ""              # ISO-8601 string
    has_attachment: bool = False
    num_links: int = 0
    thread_id: str = ""
    label: Optional[str] = None      # ground-truth category (training data only)
    id: str = ""

    def __post_init__(self) -> None:
        if self.sender and not self.sender_domain and "@" in self.sender:
            self.sender_domain = self.sender.split("@", 1)[1].lower()
        if not self.id:
            self.id = self._derive_id()

    def _derive_id(self) -> str:
        digest = hashlib.sha1(
            f"{self.sender}|{self.subject}|{self.timestamp}|{self.body[:64]}".encode()
        ).hexdigest()
        return digest[:12]

    @property
    def text(self) -> str:
        """Subject + body, the canonical text fed to NLP/ML."""
        return f"{self.subject}. {self.body}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Email":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean)


def email_text(email: "Email | dict | str") -> str:
    """Return the canonical text for an email expressed in any supported form."""
    if isinstance(email, Email):
        return email.text
    if isinstance(email, dict):
        return f"{email.get('subject', '')}. {email.get('body', '')}".strip()
    return str(email)


def as_email(email: "Email | dict | str") -> "Email":
    """Coerce a dict/str into an :class:`Email` (idempotent for Email inputs)."""
    if isinstance(email, Email):
        return email
    if isinstance(email, dict):
        return Email.from_dict(email)
    return Email(body=str(email))


# --------------------------------------------------------------------------- #
# NLP signal containers
# --------------------------------------------------------------------------- #
@dataclass
class Sentiment:
    label: str = "neutral"           # positive | neutral | negative
    score: float = 0.0               # compound score in [-1, 1]
    confidence: float = 0.0


@dataclass
class Urgency:
    level: str = "low"               # high | medium | low
    score: float = 0.0               # 0-1
    cues: list[str] = field(default_factory=list)


@dataclass
class Intent:
    label: str = "fyi"               # request | question | meeting | action_required | fyi | promotion | social
    confidence: float = 0.0


@dataclass
class NLPSignals:
    keywords: list[str] = field(default_factory=list)
    sentiment: Sentiment = field(default_factory=Sentiment)
    urgency: Urgency = field(default_factory=Urgency)
    intent: Intent = field(default_factory=Intent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": self.keywords,
            "sentiment": asdict(self.sentiment),
            "urgency": asdict(self.urgency),
            "intent": asdict(self.intent),
        }


# --------------------------------------------------------------------------- #
# Classification + priority + agent output
# --------------------------------------------------------------------------- #
@dataclass
class Classification:
    label: str = "Personal"
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class Priority:
    score: float = 0.0               # 0-100
    band: str = "Low"                # Critical | High | Medium | Low
    reasons: list[str] = field(default_factory=list)


@dataclass
class SuggestedAction:
    action: str                      # machine key, e.g. "reply"
    label: str                       # human label, e.g. "Reply now"
    reason: str = ""


@dataclass
class EmailInsight:
    """The complete agentic analysis of one email."""

    email: Email
    classification: Classification
    nlp: NLPSignals
    priority: Priority
    summary: str = ""
    suggested_actions: list[SuggestedAction] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)   # e.g. ["urgent", "vip"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "email": self.email.to_dict(),
            "classification": asdict(self.classification),
            "nlp": self.nlp.to_dict(),
            "priority": asdict(self.priority),
            "summary": self.summary,
            "suggested_actions": [asdict(a) for a in self.suggested_actions],
            "flags": self.flags,
        }
