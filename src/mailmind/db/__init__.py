"""SQLite persistence subsystem for MailMind AI.

Exposes the single public entry point, :class:`Database`, which stores emails,
insights and user-feedback actions using only the standard-library ``sqlite3``
driver.
"""
from __future__ import annotations

from .database import Database

__all__ = ["Database"]
