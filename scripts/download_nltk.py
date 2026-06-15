#!/usr/bin/env python3
"""Download the NLTK corpora MailMind depends on.

Thin runnable wrapper around :func:`mailmind.utils.text.ensure_nltk_data`. Adds
the repository's ``src`` directory to ``sys.path`` so the script works from the
repo root without installing the package, then prints the per-resource status
map::

    python scripts/download_nltk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    """Ensure NLTK data is present and print the resulting status map."""
    from mailmind.utils.text import ensure_nltk_data

    status = ensure_nltk_data(quiet=False)
    print("NLTK resource status:")
    for resource, available in sorted(status.items()):
        marker = "ok" if available else "MISSING"
        print(f"  {resource:<16} {marker}")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
