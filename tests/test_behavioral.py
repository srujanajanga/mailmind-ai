"""Tests for the behavioural-learning layer (``mailmind.behavioral``)."""
from __future__ import annotations

from mailmind.behavioral.learner import BehavioralLearner
from mailmind.schema import Email


def _email(sender: str = "alice@work.com", label: str = "Work") -> Email:
    """Build a minimal labelled email for a given sender."""
    return Email(subject="hi", body="hello", sender=sender, label=label)


def test_replied_raises_sender_engagement():
    """Recording replies pushes sender engagement positive."""
    learner = BehavioralLearner(db=None)
    email = _email()
    assert learner.sender_engagement(email.sender) == 0.0

    for _ in range(3):
        learner.record_action(email, "replied")
    assert learner.sender_engagement(email.sender) > 0.0


def test_deleted_lowers_sender_engagement():
    """Recording deletes pushes sender engagement negative."""
    learner = BehavioralLearner(db=None)
    email = _email(sender="spam@bad.com", label="Spam")
    for _ in range(3):
        learner.record_action(email, "deleted")
    assert learner.sender_engagement(email.sender) < 0.0


def test_category_engagement_tracks_actions():
    """Category engagement moves in the direction of the recorded action."""
    learner = BehavioralLearner(db=None)
    email = _email(label="Promotions")
    for _ in range(3):
        learner.record_action(email, "ignored")
    assert learner.category_engagement("Promotions") < 0.0


def test_adjust_moves_score_up_for_engaged_sender():
    """A positively-engaged sender raises the adjusted score."""
    learner = BehavioralLearner(db=None)
    email = _email()
    for _ in range(5):
        learner.record_action(email, "replied")

    base = 0.5
    adjusted, reasons = learner.adjust(base, email, "Work")
    assert adjusted > base
    assert 0.0 <= adjusted <= 1.0
    assert isinstance(reasons, list)


def test_adjust_moves_score_down_for_ignored_sender():
    """A negatively-engaged sender lowers the adjusted score."""
    learner = BehavioralLearner(db=None)
    email = _email(sender="noise@promo.com", label="Promotions")
    for _ in range(5):
        learner.record_action(email, "deleted")

    base = 0.5
    adjusted, _ = learner.adjust(base, email, "Promotions")
    assert adjusted < base
    assert 0.0 <= adjusted <= 1.0


def test_unknown_action_is_ignored():
    """An action outside VALID_ACTIONS leaves engagement untouched."""
    learner = BehavioralLearner(db=None)
    email = _email()
    learner.record_action(email, "forwarded")  # not a valid action
    assert learner.sender_engagement(email.sender) == 0.0
