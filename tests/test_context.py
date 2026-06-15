"""Tests for the context-aware priority scorer (``mailmind.context``)."""
from __future__ import annotations

from mailmind import config, nlp
from mailmind.context.scorer import ContextScorer
from mailmind.schema import Classification, Email, Priority


def test_important_vip_urgent_outranks_promotion():
    """An Important + urgent + VIP email scores higher than a low-key promo."""
    scorer = ContextScorer()

    important = Email(
        subject="URGENT: action required",
        body="Please verify your account immediately, this is time-sensitive!!!",
        sender="boss@bank.com",            # bank.com is a VIP domain
        sender_domain="bank.com",
        timestamp="2026-06-14T08:30:00",
        label="Important",
    )
    important_clf = Classification(label="Important", confidence=0.9)
    important_nlp = nlp.analyze_text(important.text, important)
    important_priority = scorer.score(important, important_clf, important_nlp)

    promo = Email(
        subject="Newsletter",
        body="Here is our monthly newsletter with some updates.",
        sender="news@shop.com",
        sender_domain="shop.com",
        timestamp="2026-06-08T10:00:00",
        label="Promotions",
    )
    promo_clf = Classification(label="Promotions", confidence=0.6)
    promo_nlp = nlp.analyze_text(promo.text, promo)
    promo_priority = scorer.score(promo, promo_clf, promo_nlp)

    assert isinstance(important_priority, Priority)
    assert important_priority.score > promo_priority.score
    assert 0.0 <= promo_priority.score <= 100.0
    assert 0.0 <= important_priority.score <= 100.0


def test_score_band_matches_config():
    """The reported band agrees with config.band_for_score."""
    scorer = ContextScorer()
    email = Email(subject="hi", body="hello", label="Personal")
    clf = Classification(label="Personal", confidence=0.5)
    signals = nlp.analyze_text(email.text, email)
    priority = scorer.score(email, clf, signals)
    assert priority.band == config.band_for_score(priority.score)


def test_vip_domain_sender_importance():
    """A VIP-domain sender gets the maximum sender-importance score."""
    scorer = ContextScorer()
    vip = Email(subject="hi", body="hello", sender="x@ceo.com", sender_domain="ceo.com")
    plain = Email(subject="hi", body="hello", sender="x@random.com", sender_domain="random.com")
    assert scorer.sender_importance(vip) == 1.0
    assert scorer.sender_importance(plain) < scorer.sender_importance(vip)
