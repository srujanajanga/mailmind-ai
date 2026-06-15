"""FastAPI application exposing the MailMind AI agent over HTTP.

Run locally with::

    uvicorn mailmind.api.main:app --reload

(or ``scripts/run_api.sh``). This module is a standalone entry point, so the web
framework dependencies (FastAPI / Pydantic / Uvicorn) are imported at the top
level. A single :class:`~mailmind.agent.agent.MailMindAgent` backed by a SQLite
:class:`~mailmind.db.database.Database` is built lazily on first use and cached
on ``app.state`` so requests share one model and one connection.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mailmind import __version__
from mailmind.schema import Email


# --------------------------------------------------------------------------- #
# Request models (Pydantic v2)
# --------------------------------------------------------------------------- #
class EmailIn(BaseModel):
    """Inbound payload describing a single email."""

    subject: str = ""
    body: str = ""
    sender: str = ""
    timestamp: str = ""
    has_attachment: bool = False
    num_links: int = 0

    def to_email(self) -> Email:
        """Convert the API payload into a domain :class:`Email`."""
        return Email(
            subject=self.subject,
            body=self.body,
            sender=self.sender,
            timestamp=self.timestamp,
            has_attachment=self.has_attachment,
            num_links=self.num_links,
        )


class FeedbackIn(BaseModel):
    """Inbound payload recording a user action on an email."""

    email_id: str
    sender: str = ""
    action: str
    category: str = ""


class InboxIn(BaseModel):
    """Inbound payload describing a batch of emails to triage."""

    emails: list[EmailIn] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def _get_agent(app: FastAPI) -> Any:
    """Return the cached agent, building it (and its DB) on first access."""
    agent = getattr(app.state, "agent", None)
    if agent is None:
        # Imported lazily so that simply importing this module does not require
        # scikit-learn / NLTK to be present at import time.
        from mailmind.agent.agent import MailMindAgent
        from mailmind.db.database import Database

        app.state.db = Database()
        agent = MailMindAgent(db=app.state.db)
        app.state.agent = agent
    return agent


def create_app() -> FastAPI:
    """Build and configure the MailMind FastAPI application."""
    app = FastAPI(
        title="MailMind AI",
        version=__version__,
        description="Classify, prioritise, summarise and act on email.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Liveness probe reporting version and the active classifier type."""
        agent = _get_agent(request.app)
        classifier = getattr(agent, "classifier", None)
        return {
            "status": "ok",
            "version": __version__,
            "classifier": type(classifier).__name__ if classifier else "unknown",
        }

    @app.post("/classify")
    def classify(payload: EmailIn, request: Request) -> dict[str, Any]:
        """Classify a single email into one of the MailMind categories."""
        from dataclasses import asdict

        agent = _get_agent(request.app)
        result = agent.classifier.classify(payload.to_email())
        return asdict(result)

    @app.post("/analyze")
    def analyze(payload: EmailIn) -> dict[str, Any]:
        """Return the NLP signals (keywords, sentiment, urgency, intent)."""
        from mailmind.nlp import analyze_text

        email = payload.to_email()
        signals = analyze_text(email.text, email=email)
        return signals.to_dict()

    @app.post("/process")
    def process(payload: EmailIn, request: Request) -> dict[str, Any]:
        """Run the full agent pipeline over one email."""
        agent = _get_agent(request.app)
        insight = agent.process_email(payload.to_email())
        return insight.to_dict()

    @app.post("/process_inbox")
    def process_inbox(payload: InboxIn, request: Request) -> list[dict[str, Any]]:
        """Run the full agent pipeline over a batch, sorted by priority."""
        agent = _get_agent(request.app)
        emails = [item.to_email() for item in payload.emails]
        insights = agent.process_inbox(emails)
        return [insight.to_dict() for insight in insights]

    @app.post("/feedback")
    def feedback(payload: FeedbackIn, request: Request) -> dict[str, Any]:
        """Record a user action so the agent can learn engagement patterns."""
        agent = _get_agent(request.app)
        email = Email(
            id=payload.email_id,
            sender=payload.sender,
            label=payload.category or None,
        )
        agent.record_feedback(email, payload.action)
        return {"ok": True}

    @app.get("/stats")
    def stats(request: Request) -> dict[str, Any]:
        """Return aggregate statistics about processed mail and feedback."""
        agent = _get_agent(request.app)
        return agent.stats()

    return app


app = create_app()
