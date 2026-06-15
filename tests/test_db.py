"""Tests for the SQLite persistence layer (``mailmind.db.database``)."""
from __future__ import annotations

from mailmind.schema import Email


def _email(sender: str = "alice@work.com", label: str = "Work") -> Email:
    """Build a minimal labelled email."""
    return Email(subject="hi", body="hello", sender=sender, label=label)


def test_record_action_increments_sender_and_category(tmp_db):
    """record_action bumps both the per-sender and per-category tallies."""
    email = _email()
    tmp_db.record_action(
        email_id=email.id,
        sender=email.sender,
        category=email.label,
        action="replied",
    )

    sender_counts = tmp_db.sender_action_counts(email.sender)
    category_counts = tmp_db.category_action_counts(email.label)
    assert sender_counts["replied"] == 1
    assert category_counts["replied"] == 1
    # Untouched actions remain zero.
    assert sender_counts["deleted"] == 0


def test_invalid_action_is_ignored(tmp_db):
    """An action outside VALID_ACTIONS is silently dropped."""
    tmp_db.record_action(
        email_id="abc", sender="x@y.com", category="Work", action="forwarded"
    )
    assert tmp_db.action_counts() == {
        "opened": 0,
        "replied": 0,
        "ignored": 0,
        "deleted": 0,
    }


def test_action_counts_aggregates_globally(tmp_db):
    """Global action_counts aggregate across all logged actions."""
    tmp_db.record_action(email_id="1", sender="a@x.com", category="Work", action="opened")
    tmp_db.record_action(email_id="2", sender="b@x.com", category="Work", action="opened")
    tmp_db.record_action(email_id="3", sender="c@x.com", category="Spam", action="deleted")

    counts = tmp_db.action_counts()
    assert counts["opened"] == 2
    assert counts["deleted"] == 1


def test_total_emails_and_save(tmp_db):
    """save_email persists rows and total_emails counts them (dedup by id)."""
    assert tmp_db.total_emails() == 0
    email = _email()
    tmp_db.save_email(email)
    tmp_db.save_email(email)  # same id -> upsert, not a new row
    assert tmp_db.total_emails() == 1


def test_reset_clears_everything(tmp_db):
    """reset wipes emails and action tallies."""
    email = _email()
    tmp_db.save_email(email)
    tmp_db.record_action(
        email_id=email.id, sender=email.sender, category=email.label, action="opened"
    )

    tmp_db.reset()
    assert tmp_db.total_emails() == 0
    assert tmp_db.action_counts() == {
        "opened": 0,
        "replied": 0,
        "ignored": 0,
        "deleted": 0,
    }
    assert tmp_db.sender_action_counts(email.sender)["opened"] == 0


def test_context_manager_closes(tmp_path):
    """The Database supports the context-manager protocol."""
    from mailmind.db.database import Database

    with Database(tmp_path / "ctx.db") as db:
        db.save_email(_email())
        assert db.total_emails() == 1
