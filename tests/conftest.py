"""Shared pytest configuration and fixtures for the MailMind AI test-suite.

This module performs two jobs:

* It inserts the repository's ``src`` directory onto :data:`sys.path` so the
  ``mailmind`` package imports cleanly when the suite is run with a bare
  ``pytest`` from the repository root (no ``pip install`` required).
* It defines small, deterministic, session-friendly fixtures shared across the
  test modules: a tiny generated dataset, sample :class:`Email` objects (one per
  category), a temporary :class:`Database`, and a lightweight trained
  :class:`MailMindClassifier`.

Every fixture deliberately uses tiny sample sizes so the whole suite stays fast
and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Make the ``src`` layout importable without installation.
# --------------------------------------------------------------------------- #
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mailmind import config  # noqa: E402  (import after sys.path mutation)
from mailmind.data import generate_dataset  # noqa: E402
from mailmind.db.database import Database  # noqa: E402
from mailmind.ml.classifier import MailMindClassifier  # noqa: E402
from mailmind.schema import Email  # noqa: E402

# Number of rows per category used by the lightweight training fixtures. Kept
# small so the suite runs quickly while still giving every category enough
# signal to be learned.
TRAIN_PER_CATEGORY = 20


# --------------------------------------------------------------------------- #
# Dataset / DataFrame fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def small_dataset():
    """Return a tiny balanced dataset (5 rows per category) as a DataFrame."""
    return generate_dataset(samples_per_category=5, seed=config.RANDOM_SEED)


# --------------------------------------------------------------------------- #
# Sample emails (one per category)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def sample_emails() -> dict[str, Email]:
    """Return one hand-written :class:`Email` per category, keyed by label.

    The bodies are crafted so the heuristic classifier and the NLP layer have
    clear cues to react to, keeping assertions readable and robust.
    """
    return {
        "Important": Email(
            subject="URGENT: action required on your account",
            body="Please verify your account immediately or it will be suspended.",
            sender="alerts@bank.com",
            sender_name="Security Team",
            sender_domain="bank.com",
            timestamp="2026-06-14T08:30:00",
            label="Important",
        ),
        "Work": Email(
            subject="Project review meeting agenda",
            body="Let's sync on the quarterly report and the client deliverable schedule.",
            sender="alice@work.com",
            sender_name="Alice Smith",
            sender_domain="work.com",
            timestamp="2026-06-14T07:00:00",
            label="Work",
        ),
        "Personal": Email(
            subject="Dinner this weekend?",
            body="Hi! Mom asked if you want to come home for a family dinner on Sunday.",
            sender="jane@gmail.com",
            sender_name="Jane Doe",
            sender_domain="gmail.com",
            timestamp="2026-06-13T19:00:00",
            label="Personal",
        ),
        "Social": Email(
            subject="Someone liked your post",
            body="You were tagged in a comment and have a new friend request to review.",
            sender="notify@social.com",
            sender_name="SocialNet",
            sender_domain="social.com",
            timestamp="2026-06-13T12:00:00",
            label="Social",
        ),
        "Promotions": Email(
            subject="50% off sale ends today",
            body="Exclusive offer: get a discount with this coupon. Shop now and save!",
            sender="deals@shop.com",
            sender_name="ShopMart",
            sender_domain="shop.com",
            timestamp="2026-06-12T10:00:00",
            num_links=2,
            label="Promotions",
        ),
        "Spam": Email(
            subject="Congratulations you won a prize",
            body="Claim your free cash lottery winnings now! Guaranteed risk-free prize.",
            sender="winner@lotto.biz",
            sender_name="Lotto",
            sender_domain="lotto.biz",
            timestamp="2026-06-11T03:00:00",
            label="Spam",
        ),
    }


@pytest.fixture
def sample_email(sample_emails) -> Email:
    """Return a single representative :class:`Email` (the Work sample)."""
    return sample_emails["Work"]


# --------------------------------------------------------------------------- #
# Temporary database
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_db(tmp_path) -> Database:
    """Yield a fresh :class:`Database` backed by a temp file, closed on teardown."""
    db = Database(tmp_path / "mailmind_test.db")
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Lightweight trained classifier (session-scoped: trained exactly once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def trained_classifier() -> MailMindClassifier:
    """Return a :class:`MailMindClassifier` trained on a small generated corpus.

    Training happens once per test session on ``TRAIN_PER_CATEGORY`` rows per
    category, which is enough for the model to fit and predict while keeping the
    suite fast.
    """
    frame = generate_dataset(
        samples_per_category=TRAIN_PER_CATEGORY, seed=config.RANDOM_SEED
    )
    emails = frame.to_dict(orient="records")
    labels = [row["label"] for row in emails]
    return MailMindClassifier().fit(emails, labels)
