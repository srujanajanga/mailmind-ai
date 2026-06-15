#!/usr/bin/env python3
"""Generate the synthetic MailMind email dataset.

Thin runnable wrapper around :func:`mailmind.data.dataset_generator.main`. Adds
the repository's ``src`` directory to ``sys.path`` so the script works from the
repo root without installing the package::

    python scripts/generate_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    """Invoke the dataset generator and return its exit status."""
    from mailmind.data.dataset_generator import main as generate

    result = generate()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
