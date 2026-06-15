"""SQLite-backed persistence for MailMind AI.

The :class:`Database` class is the single gateway to durable storage. It stores
raw emails, the agent's enriched insights, an append-only log of user feedback
actions, and denormalised per-sender / per-category action tallies that the
behavioural-learning layer reads to estimate engagement.

Only the Python standard-library :mod:`sqlite3` driver is used. Connections are
opened with ``check_same_thread=False`` and a :class:`sqlite3.Row` row factory so
results behave like dictionaries and the same instance can be shared loosely
across threads (callers remain responsible for not issuing concurrent writes).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import config
from ..schema import Email, EmailInsight, as_email
from .models import DDL_STATEMENTS, TABLE_NAMES, row_to_dict

# The four canonical action columns shared by sender_stats and category_stats.
_ACTION_COLUMNS: tuple[str, ...] = ("opened", "replied", "ignored", "deleted")


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string (storage timestamps)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _zero_counts() -> dict[str, int]:
    """Return a fresh ``{action: 0}`` mapping for the four tracked actions."""
    return {column: 0 for column in _ACTION_COLUMNS}


class Database:
    """A thin, well-typed wrapper around a SQLite database file.

    Instances can be used as context managers; the connection is closed on exit::

        with Database(path) as db:
            db.save_email(email)
    """

    def __init__(self, path: str | Path = config.DB_PATH) -> None:
        """Open (or create) the database at ``path`` and ensure the schema exists.

        Parameters
        ----------
        path:
            Filesystem location of the SQLite file. The parent directory is
            created if it does not already exist.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _create_tables(self) -> None:
        """Create every table defined in :mod:`mailmind.db.models` if missing."""
        with self._conn:
            for statement in DDL_STATEMENTS:
                self._conn.execute(statement)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def save_email(self, email: Email | dict | str) -> None:
        """Insert or update a single email keyed by its stable id.

        Accepts an :class:`Email`, a ``dict`` or a raw string; non-``Email``
        inputs are coerced via :func:`mailmind.schema.as_email`.
        """
        mail = as_email(email)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO emails (
                    id, sender, sender_name, sender_domain, subject, body,
                    timestamp, has_attachment, num_links, label, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    sender        = excluded.sender,
                    sender_name   = excluded.sender_name,
                    sender_domain = excluded.sender_domain,
                    subject       = excluded.subject,
                    body          = excluded.body,
                    timestamp     = excluded.timestamp,
                    has_attachment= excluded.has_attachment,
                    num_links     = excluded.num_links,
                    label         = excluded.label
                """,
                (
                    mail.id,
                    mail.sender,
                    mail.sender_name,
                    mail.sender_domain,
                    mail.subject,
                    mail.body,
                    mail.timestamp,
                    int(bool(mail.has_attachment)),
                    int(mail.num_links or 0),
                    mail.label,
                    _utc_now(),
                ),
            )

    def save_insight(self, insight: EmailInsight) -> None:
        """Persist one :class:`EmailInsight` as a flattened ``insights`` row.

        Nested signal objects are reduced to their primary fields: urgency to its
        level, sentiment and intent to their labels, and flags to a comma-joined
        string.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO insights (
                    email_id, category, confidence, priority_score, priority_band,
                    urgency, sentiment, intent, summary, flags, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    insight.email.id,
                    insight.classification.label,
                    float(insight.classification.confidence),
                    float(insight.priority.score),
                    insight.priority.band,
                    insight.nlp.urgency.level,
                    insight.nlp.sentiment.label,
                    insight.nlp.intent.label,
                    insight.summary,
                    ",".join(insight.flags),
                    _utc_now(),
                ),
            )

    def record_action(
        self,
        *,
        email_id: str,
        sender: str,
        category: str,
        action: str,
        ts: str = "",
    ) -> None:
        """Log a user feedback action and update the rolled-up stat tables.

        Invalid actions (anything not in :data:`config.VALID_ACTIONS`) are
        silently ignored so callers need not pre-validate. The matching column in
        both ``sender_stats`` and ``category_stats`` is incremented atomically.
        """
        if action not in config.VALID_ACTIONS:
            return
        timestamp = ts or _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO actions (email_id, sender, category, action, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email_id, sender, category, action, timestamp),
            )
            self._bump_stat("sender_stats", "sender", sender, action)
            self._bump_stat("category_stats", "category", category, action)

    def _bump_stat(self, table: str, key_column: str, key: str, action: str) -> None:
        """Increment ``action`` count for ``key`` in a stats table.

        Uses ``INSERT OR IGNORE`` to guarantee the row exists, then a targeted
        ``UPDATE``. ``table``, ``key_column`` and ``action`` are module-controlled
        identifiers (never user input), so interpolation here is safe.
        """
        if action not in _ACTION_COLUMNS or not key:
            return
        self._conn.execute(
            f"INSERT OR IGNORE INTO {table} ({key_column}) VALUES (?)",
            (key,),
        )
        self._conn.execute(
            f"UPDATE {table} SET {action} = {action} + 1 WHERE {key_column} = ?",
            (key,),
        )

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def sender_action_counts(self, sender: str) -> dict[str, int]:
        """Return the ``{action: count}`` tally for one sender (zeros if unseen)."""
        return self._stat_counts("sender_stats", "sender", sender)

    def category_action_counts(self, category: str) -> dict[str, int]:
        """Return the ``{action: count}`` tally for one category (zeros if unseen)."""
        return self._stat_counts("category_stats", "category", category)

    def _stat_counts(self, table: str, key_column: str, key: str) -> dict[str, int]:
        """Fetch the four action counts for ``key`` from a stats table."""
        counts = _zero_counts()
        row = self._conn.execute(
            f"SELECT opened, replied, ignored, deleted "
            f"FROM {table} WHERE {key_column} = ?",
            (key,),
        ).fetchone()
        if row is not None:
            for column in _ACTION_COLUMNS:
                counts[column] = int(row[column] or 0)
        return counts

    def action_counts(self) -> dict[str, int]:
        """Return global counts of every action type across the whole log."""
        counts = _zero_counts()
        rows = self._conn.execute(
            "SELECT action, COUNT(*) AS n FROM actions GROUP BY action"
        ).fetchall()
        for row in rows:
            action = row["action"]
            if action in counts:
                counts[action] = int(row["n"])
        return counts

    def total_emails(self) -> int:
        """Return the number of stored emails."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM emails").fetchone()
        return int(row["n"]) if row is not None else 0

    def recent_insights(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent insights joined with email subject and sender.

        Results are ordered by insight ``created_at`` descending and capped at
        ``limit`` rows. Each entry is a plain ``dict`` carrying the insight fields
        plus the originating email's ``subject`` and ``sender``.
        """
        rows = self._conn.execute(
            """
            SELECT
                i.email_id, i.category, i.confidence, i.priority_score,
                i.priority_band, i.urgency, i.sentiment, i.intent, i.summary,
                i.flags, i.created_at,
                e.subject AS subject, e.sender AS sender
            FROM insights AS i
            LEFT JOIN emails AS e ON e.id = i.email_id
            ORDER BY i.created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Maintenance / lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Delete every row from all tables, leaving the schema intact."""
        with self._conn:
            for table in TABLE_NAMES:
                self._conn.execute(f"DELETE FROM {table}")

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
