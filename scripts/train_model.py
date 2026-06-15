#!/usr/bin/env python3
"""Train and persist the MailMind classifier.

Thin runnable wrapper around :func:`mailmind.ml.train.main`. Adds the
repository's ``src`` directory to ``sys.path`` so the script works from the repo
root without installing the package::

    python scripts/train_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    """Invoke the training routine and return its exit status."""
    from mailmind.ml.train import main as train

    result = train()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
