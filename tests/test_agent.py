"""Tests for the orchestrating agent (``mailmind.agent.agent``)."""
from __future__ import annotations

from mailmind import config
from mailmind.agent.agent import MailMindAgent
from mailmind.ml.classifier import HeuristicClassifier
from mailmind.schema import Classification, Email, EmailInsight, NLPSignals, Priority


def _heuristic_agent(db=None) -> MailMindAgent:
    """Build an agent backed by the heuristic classifier (no model file needed)."""
    return MailMindAgent(classifier=HeuristicClassifier(), db=db)


def test_process_email_populates_insight(sample_emails):
    """process_email returns a fully-populated EmailInsight."""
    agent = _heuristic_agent()
    insight = agent.process_email(sample_emails["Work"])

    assert isinstance(insight, EmailInsight)
    assert isinstance(insight.email, Email)
    assert isinstance(insight.classification, Classification)
    assert isinstance(insight.nlp, NLPSignals)
    assert isinstance(insight.priority, Priority)
    assert insight.classification.label in config.CATEGORIES
    assert insight.priority.band in {"Critical", "High", "Medium", "Low"}
    assert isinstance(insight.summary, str)
    assert isinstance(insight.suggested_actions, list)
    assert isinstance(insight.flags, list)


def test_process_email_accepts_raw_inputs():
    """A plain dict / string flows through the pipeline without error."""
    agent = _heuristic_agent()
    insight = agent.process_email({"subject": "sale", "body": "50% off, shop now"})
    assert isinstance(insight, EmailInsight)


def test_process_inbox_sorted_by_priority(sample_emails):
    """process_inbox returns insights sorted by priority score (desc)."""
    agent = _heuristic_agent()
    insights = agent.process_inbox(list(sample_emails.values()))

    assert len(insights) == len(sample_emails)
    scores = [i.priority.score for i in insights]
    assert scores == sorted(scores, reverse=True)


def test_record_feedback_runs(sample_emails):
    """record_feedback completes without raising."""
    agent = _heuristic_agent()
    email = sample_emails["Work"]
    agent.record_feedback(email, "replied")
    # Engagement should now be positive for that sender.
    assert agent.behavioral.sender_engagement(email.sender) > 0.0


def test_agent_uses_heuristic_fallback(tmp_path, sample_email):
    """With no model on disk the agent degrades to the heuristic classifier."""
    missing_model = tmp_path / "does_not_exist.joblib"
    agent = MailMindAgent(model_path=missing_model)
    assert type(agent.classifier).__name__ == "HeuristicClassifier"

    insight = agent.process_email(sample_email)
    assert isinstance(insight, EmailInsight)


def test_agent_persists_with_db(tmp_db, sample_emails):
    """When a database is attached, processing persists email + insight."""
    agent = _heuristic_agent(db=tmp_db)
    agent.process_email(sample_emails["Important"])

    assert tmp_db.total_emails() == 1
    assert len(tmp_db.recent_insights(limit=10)) == 1


def test_stats_reports_classifier_name():
    """stats names the active classifier and returns the expected shape."""
    agent = _heuristic_agent()
    stats = agent.stats()
    assert stats["classifier"] == "HeuristicClassifier"
    assert "actions" in stats
    assert "total_emails" in stats
