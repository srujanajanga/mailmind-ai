"""SQLite schema definitions and row helpers for the MailMind persistence layer.

This module holds the raw DDL for every table the :class:`mailmind.db.database.Database`
manages, plus a tiny helper to coerce :class:`sqlite3.Row` objects into plain
dictionaries. Keeping the schema here (rather than inline in the database class)
makes the storage contract easy to read and to evolve in one place.
"""
from __future__ import annotations

import sqlite3
from typing import Any

# --------------------------------------------------------------------------- #
# Table DDL
# --------------------------------------------------------------------------- #
# Raw emails as received by the agent. ``id`` is the stable Email.id digest so
# repeated saves of the same message upsert rather than duplicate.
EMAILS_DDL: str = """
CREATE TABLE IF NOT EXISTS emails (
    id            TEXT PRIMARY KEY,
    sender        TEXT,
    sender_name   TEXT,
    sender_domain TEXT,
    subject       TEXT,
    body          TEXT,
    timestamp     TEXT,
    has_attachment INT,
    num_links     INT,
    label         TEXT,
    created_at    TEXT
)
"""

# One enriched analysis row per processed email. Flags are stored comma-joined.
INSIGHTS_DDL: str = """
CREATE TABLE IF NOT EXISTS insights (
    email_id       TEXT,
    category       TEXT,
    confidence     REAL,
    priority_score REAL,
    priority_band  TEXT,
    urgency        TEXT,
    sentiment      TEXT,
    intent         TEXT,
    summary        TEXT,
    flags          TEXT,
    created_at     TEXT
)
"""

# Append-only log of user feedback actions (replied/opened/ignored/deleted).
ACTIONS_DDL: str = """
CREATE TABLE IF NOT EXISTS actions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id TEXT,
    sender   TEXT,
    category TEXT,
    action   TEXT,
    ts       TEXT
)
"""

# Per-sender rolled-up action counts (denormalised for fast engagement reads).
SENDER_STATS_DDL: str = """
CREATE TABLE IF NOT EXISTS sender_stats (
    sender  TEXT PRIMARY KEY,
    opened  INT DEFAULT 0,
    replied INT DEFAULT 0,
    ignored INT DEFAULT 0,
    deleted INT DEFAULT 0
)
"""

# Per-category rolled-up action counts.
CATEGORY_STATS_DDL: str = """
CREATE TABLE IF NOT EXISTS category_stats (
    category TEXT PRIMARY KEY,
    opened   INT DEFAULT 0,
    replied  INT DEFAULT 0,
    ignored  INT DEFAULT 0,
    deleted  INT DEFAULT 0
)
"""

# Executed in order on every connection; ``IF NOT EXISTS`` makes this idempotent.
DDL_STATEMENTS: list[str] = [
    EMAILS_DDL,
    INSIGHTS_DDL,
    ACTIONS_DDL,
    SENDER_STATS_DDL,
    CATEGORY_STATS_DDL,
]

# Tables wiped by :meth:`Database.reset`, in an order safe for deletion.
TABLE_NAMES: tuple[str, ...] = (
    "emails",
    "insights",
    "actions",
    "sender_stats",
    "category_stats",
)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    """Convert a :class:`sqlite3.Row` into a plain ``dict``.

    Returns an empty dict when ``row`` is ``None`` so callers can treat a missing
    record uniformly without a separate null check.
    """
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}
